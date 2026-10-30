"""
Train the deployment model on ALL available years (2025 included) and calibrate
its threshold by cross-fitting instead of by holding a year out.

calibrate_threshold.py held 2025 out of training purely so its wet season could
serve as a clean threshold-calibration set. That costs almost no skill
(-0.0005 AUC) but it does throw away the most recent monsoon, and it calibrates
the threshold on a single year.

Cross-fitting gets both. For each ECMWF year Y, a fold model is trained WITHOUT
Y and used to score Y's wet season; those out-of-fold probabilities are pooled
across all years and the threshold is picked on the pool. The final model is
then trained on every year. The threshold is never fit on data its own model
saw, and it is averaged over several monsoons instead of one.

Usage:
  python calibrate_threshold_crossfit.py --root_dir <PREP> --out_dir <...> \
      --calib_years 2023,2024,2025 --seed 1
"""

import os
import gc
import random
import json
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, roc_auc_score, average_precision_score,
                             precision_score, recall_score)

from calibrate_threshold import build, thr_grid, best_f1_threshold, metrics_at, HP

BASE_UKMO = 'UKMOaffine:2016-2022'


def get_args():
    p = argparse.ArgumentParser(description='Cross-fitted threshold, model trained on all years')
    p.add_argument('--root_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--calib_years', default='2023,2024,2025',
                   help='ECMWF years to cross-fit over (each is held out in turn)')
    p.add_argument('--ecmwf_span', default='2023-2025', help='ECMWF years in the final model')
    p.add_argument('--calib_window', default='07-01:10-31')
    p.add_argument('--test_year', default='2026')
    p.add_argument('--seed', default=1, type=int)
    p.add_argument('--hindcast', default='GPMv07',
                   help="LabelledData_<hindcast> tree to read, e.g. GPMv07_latfix for the "
                        "latitude-corrected forecast features")
    p.add_argument('--max_depth', default=None,
                   help="Override HP max_depth (e.g. 10 for a far smaller pickle); default keeps None")
    return p.parse_args()


def sources_excluding(year, ecmwf_span):
    """Config D source spec with one ECMWF year dropped."""
    a, b = [int(v) for v in ecmwf_span.split('-')]
    spans = ['ecmwf:{}-{}'.format(y, y) for y in range(a, b + 1) if str(y) != str(year)]
    return ','.join([BASE_UKMO] + spans)


def fit(Xfull, y, feats, hp, seed):
    """
    run_classifier.load_data shuffles with an unseeded RNG, so the row order of
    the training frame differs run to run. RandomForest's bootstrap draws by
    position, so that alone changes the fitted model - two identical-seed runs
    produced thresholds 0.193 and 0.230. Sorting on (date, district) pins it.
    Per-year index values repeat across the concatenated years, so the frame's
    own index cannot be used as the key.
    """
    o = np.lexsort((Xfull['district'].to_numpy(), Xfull['date'].to_numpy()))
    X, y = Xfull.iloc[o][feats], np.asarray(y)[o]
    m = RandomForestClassifier(criterion='gini', random_state=seed, n_jobs=-1, **hp)
    t = time.time()
    m.fit(X, y)
    print('    fit {} rows in {:.0f}s'.format(len(X), time.time() - t), flush=True)
    return m


if __name__ == '__main__':
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    root, out = args.root_dir, args.out_dir
    hind, ens, fm = args.hindcast, '0', 'UKMO'
    os.makedirs(out, exist_ok=True)
    info = ['date', 'district']
    md_a, md_b = args.calib_window.split(':')
    hp = dict(HP)
    if args.max_depth is not None:
        hp['max_depth'] = None if args.max_depth.lower() == 'none' else int(args.max_depth)
    all_sources = '{},ecmwf:{}'.format(BASE_UKMO, args.ecmwf_span)
    calib_years = args.calib_years.split(',')

    # ---- Cross-fitted out-of-fold probabilities on each year's wet season
    pool, full, per_year = [], [], []
    for yr in calib_years:
        src = sources_excluding(yr, args.ecmwf_span)
        print('\n=== fold {}: train [{}] -> score {} wet season'.format(yr, src, yr), flush=True)
        Xtr, ytr, Xcal, ycal, Xva, yva = build(root, hind, ens, fm, src, yr)
        feats = [c for c in Xtr.columns if c not in info]
        wet = (Xcal['date'] >= '{}-{}'.format(yr, md_a)) & (Xcal['date'] <= '{}-{}'.format(yr, md_b))
        Xw, yw = Xcal[wet], ycal[wet]
        m = fit(pd.concat([Xtr, Xva]), pd.concat([ytr, yva]), feats, hp, args.seed)
        # Score the whole season for the plots, but calibrate on the wet window only.
        p_full = m.predict_proba(Xcal[feats])[:, 1]
        full.append(pd.DataFrame({'model soft predictions': p_full, 'groundtruth': ycal.values,
                                  'date': Xcal['date'].values, 'district': Xcal['district'].values,
                                  'year': yr}))
        p = p_full[wet.values]
        t_y, f_y = best_f1_threshold(yw.values, p)
        print('    {} wet: {} rows {} -> {}, {:.1%} positive'.format(
            yr, len(Xw), Xw['date'].min(), Xw['date'].max(), yw.mean()), flush=True)
        print('    year-specific best thr {:.4f} (F1 {:.4f}), AUC {:.4f} AP {:.4f}'.format(
            t_y, f_y, roc_auc_score(yw, p), average_precision_score(yw, p)), flush=True)
        per_year.append({'year': yr, 'n': int(len(Xw)), 'positive_rate': float(yw.mean()),
                         'best_threshold': t_y, 'best_f1': f_y,
                         'auc': float(roc_auc_score(yw, p)), 'ap': float(average_precision_score(yw, p))})
        pool.append(pd.DataFrame({'model soft predictions': p, 'groundtruth': yw.values,
                                  'date': Xw['date'].values, 'district': Xw['district'].values,
                                  'year': yr}))
        del m, Xtr, Xva
        gc.collect()

    oof = pd.concat(pool, ignore_index=True)
    thr, f1_pool = best_f1_threshold(oof['groundtruth'].values, oof['model soft predictions'].values)
    print('\n=== CROSS-FITTED THRESHOLD = {:.4f}  (pooled F1 {:.4f} over {} rows, {} years)'.format(
        thr, f1_pool, len(oof), len(calib_years)), flush=True)
    for r in per_year:
        mm = metrics_at(oof[oof.year == r['year']]['groundtruth'].values, oof[oof.year == r['year']]['model soft predictions'].values, thr)
        r['f1_at_pooled_thr'] = mm['f1']
        r['flagged_pct_at_pooled_thr'] = mm['flagged_pct']
        r['recall_at_pooled_thr'] = mm['recall']
        print('    {}: at pooled thr F1 {:.3f}, flags {:.1f}%, recall {:.3f}  (its own best thr {:.4f})'.format(
            r['year'], mm['f1'], mm['flagged_pct'], mm['recall'], r['best_threshold']), flush=True)
    oof.to_csv('{}/crossfit_oof_predictions.csv'.format(out), index=False)
    # One run-folder per year so Analysis/analyze_run.py can plot the honest timeseries
    from sklearn.metrics import precision_recall_curve
    for yr, g in pd.concat(full, ignore_index=True).groupby('year'):
        d = '{}/oof_{}'.format(out, yr)
        os.makedirs(d, exist_ok=True)
        g.drop(columns=['year']).to_csv('{}/predictions_and_groundtruth.csv'.format(d), index=False)
        pr, rc, ts = precision_recall_curve(g['groundtruth'], g['model soft predictions'])
        pd.DataFrame({'thresholds': np.append(ts, 1.0), 'precision': pr, 'recall': rc}).to_csv(
            '{}/precision_recall.csv'.format(d), index=False)
    pd.DataFrame(per_year).to_csv('{}/crossfit_per_year.csv'.format(out), index=False)

    # ---- Final model: every year, 2025 included
    print('\n=== FINAL MODEL: train [{}] -> test {}'.format(all_sources, args.test_year), flush=True)
    Xtr, ytr, Xte, yte, Xva, yva = build(root, hind, ens, fm, all_sources, args.test_year)
    feats = [c for c in Xtr.columns if c not in info]
    final = fit(pd.concat([Xtr, Xva]), pd.concat([ytr, yva]), feats, hp, args.seed)
    p_te = final.predict_proba(Xte[feats])[:, 1]
    auc, ap = float(roc_auc_score(yte, p_te)), float(average_precision_score(yte, p_te))
    m_thr = metrics_at(yte, p_te, thr)
    thr_te, f1_te = best_f1_threshold(yte.values, p_te)
    print('  {} window {} -> {}, {} rows, {:.1%} positive'.format(
        args.test_year, Xte['date'].min(), Xte['date'].max(), len(Xte), yte.mean()), flush=True)
    print('  AUC {:.4f}  AP {:.4f}'.format(auc, ap), flush=True)
    print('  at cross-fitted thr {:.4f}: {}   <- HONEST'.format(
        thr, {k: round(v, 3) for k, v in m_thr.items()}), flush=True)
    print('  at {}-optimal thr {:.4f}: F1 {:.4f}   <- optimistic ceiling'.format(
        args.test_year, thr_te, f1_te), flush=True)

    meta = {
        'selected_by': 'crossfit_wet_season_calibration',
        'hyperparameters': hp,
        'train_sources': all_sources,
        'threshold': thr,
        'threshold_calibrated_on': 'out-of-fold {} wet seasons ({} to {}), pooled'.format(
            '+'.join(calib_years), md_a, md_b),
        'calibration_f1_pooled': f1_pool,
        'per_year_calibration': per_year,
        'honest_{}_estimate'.format(args.test_year): {
            'window': '{} to {}'.format(Xte['date'].min(), Xte['date'].max()),
            'auc': auc, 'ap': ap, **{k: float(v) for k, v in m_thr.items()}},
        'optimistic_ceiling_thr_fit_on_test': {'threshold': thr_te, 'f1': f1_te},
        'seed': args.seed,
        'hindcast_tree': hind,
        'features': feats,
        'feature_prefix_note': 'ECMWF samples must be renamed ecmwf_ens_0 -> UKMO_ens_0',
    }
    stem = '{}/rf_model_crossfit_calibrated'.format(out)
    joblib.dump(final, stem + '.joblib')
    with open(stem + '.pkl', 'wb') as f:
        pickle.dump(final, f)
    with open(stem + '_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print('\nSaved {}.pkl (threshold {:.4f})'.format(stem, thr), flush=True)

    rows = []
    for t in sorted(set([0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, round(thr, 4)])):
        rc = metrics_at(oof['groundtruth'].values, oof['model soft predictions'].values, t)
        rd = metrics_at(yte, p_te, t)
        rows.append({'threshold': round(float(t), 4),
                     'oof_wet_f1': round(rc['f1'], 3), 'oof_wet_flagged%': round(rc['flagged_pct'], 1),
                     'oof_wet_recall': round(rc['recall'], 3),
                     '{}_f1'.format(args.test_year): round(rd['f1'], 3),
                     '{}_flagged%'.format(args.test_year): round(rd['flagged_pct'], 1),
                     '{}_recall'.format(args.test_year): round(rd['recall'], 3)})
    tbl = pd.DataFrame(rows)
    tbl.to_csv('{}/threshold_sensitivity_crossfit.csv'.format(out), index=False)
    print('\nTHRESHOLD SENSITIVITY (pooled out-of-fold wet vs {})'.format(args.test_year), flush=True)
    print(tbl.to_string(index=False), flush=True)
    print('\nDone -> {}'.format(out), flush=True)
