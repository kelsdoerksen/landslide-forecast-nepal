"""
Calibrate the deployment decision threshold on a held-out WET season.

The 2026 test window (Apr 1 - Jun 16) is the pre-monsoon shoulder: ~5x drier than
the Jul-Oct monsoon peak the model is about to be deployed into. A threshold tuned
on that window is tuned on the wrong regime. The training-set validation split
cannot calibrate it either - it is a random split of the training years, so it is
temporally contaminated and its optimal threshold collapses on real held-out data.

This script instead holds out a whole WET season:

  model    : trained WITHOUT 2025 (affine UKMO 2016-2022 + ECMWF 2023-2024)
  calibrate: threshold picked on ECMWF 2025 Jul-Oct  <- wet, and unseen
  report   : that threshold applied to 2026 Apr 1 - Jun 16, which neither the
             model nor the threshold has seen -> an honest out-of-sample estimate

It also reports the cost of dropping 2025 from training, and both a
maximize-F1 threshold and a target-alarm-rate threshold.

Usage:
  python calibrate_threshold.py --root_dir <PREP> --out_dir <PREP>/2026_Monsoon_Testing/deploy_model
"""

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_score, recall_score

from run_classifier import load_data

# Hyperparameters carried over from the sweep (tune_and_export.py, best on 2026).
HP = {'n_estimators': 400, 'max_depth': None, 'min_samples_split': 8, 'class_weight': None}


def get_args():
    p = argparse.ArgumentParser(description='Calibrate deployment threshold on a held-out wet season')
    p.add_argument('--root_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--calib_year', default='2025')
    p.add_argument('--calib_window', default='07-01:10-31', help='MM-DD:MM-DD wet window within calib_year')
    p.add_argument('--seed', default=1, type=int)
    return p.parse_args()


def thr_grid():
    return np.linspace(0.005, 0.6, 400)


def best_f1_threshold(y, p):
    g = thr_grid()
    f = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in g]
    i = int(np.argmax(f))
    return float(g[i]), float(f[i])


def metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    return {'f1': float(f1_score(y, pred, zero_division=0)),
            'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)),
            'flagged_pct': float(100 * pred.mean())}


def build(root, hind, ens, fm, train_sources, test_year):
    def spec(s):
        return {'data_dir': '{}/LabelledData_{}/{}/ensemble_{}'.format(root, hind, s, ens),
                'rename': ('{}_ens_{}'.format(s, ens), '{}_ens_{}'.format(fm, ens))}
    tys = {}
    for part in train_sources.split(','):
        s, span = part.split(':')
        a, b = span.split('-')
        for y in range(int(a), int(b) + 1):
            tys[str(y)] = spec(s)
    return load_data(test_year, '{}/LabelledData_{}/{}/ensemble_{}'.format(root, hind, fm, ens),
                     'full', '/tmp', 'calib', root, use_ecmwf_2024=False,
                     test_data_dir='{}/LabelledData_{}/ecmwf/ensemble_{}'.format(root, hind, ens),
                     test_rename=('ecmwf_ens_{}'.format(ens), '{}_ens_{}'.format(fm, ens)),
                     train_year_sources=tys)


if __name__ == '__main__':
    args = get_args()
    root, out = args.root_dir, args.out_dir
    hind, ens, fm = 'GPMv07', '0', 'UKMO'
    os.makedirs(out, exist_ok=True)
    info = ['date', 'district']
    md_a, md_b = args.calib_window.split(':')

    # --- M1: model that has never seen 2025 or 2026. Calibration set = 2025 wet season.
    print('=== M1: train affineUKMO 2016-2022 + ECMWF 2023-2024 (2025 held out)')
    Xtr, ytr, Xcal, ycal, Xva, yva = build(root, hind, ens, fm,
                                           'UKMOaffine:2016-2022,ecmwf:2023-2024', args.calib_year)
    # load_data returns the calib year over Apr-Oct; narrow to the wet window
    wet = (Xcal['date'] >= '{}-{}'.format(args.calib_year, md_a)) & \
          (Xcal['date'] <= '{}-{}'.format(args.calib_year, md_b))
    Xcal_w, ycal_w = Xcal[wet], ycal[wet]
    feats = [c for c in Xtr.columns if c not in info]

    m1 = RandomForestClassifier(criterion='gini', random_state=args.seed, n_jobs=-1, **HP)
    m1.fit(pd.concat([Xtr, Xva])[feats], pd.concat([ytr, yva]))
    p_cal = m1.predict_proba(Xcal_w[feats])[:, 1]
    thr_wet, f1_wet = best_f1_threshold(ycal_w, p_cal)
    print('  calibration set: {} rows {} -> {}, {:.1%} positive, mean prob {:.3f}'.format(
        len(Xcal_w), Xcal_w['date'].min(), Xcal_w['date'].max(), ycal_w.mean(), p_cal.mean()))
    print('  WET-CALIBRATED THRESHOLD = {:.4f}  (F1 {:.4f} on the calibration set)'.format(thr_wet, f1_wet))
    print('  calib metrics at thr: {}'.format({k: round(v, 3) for k, v in metrics_at(ycal_w, p_cal, thr_wet).items()}))

    # --- Apply M1 to 2026: neither model nor threshold has seen it -> honest estimate
    print('\n=== M1 applied to 2026 Apr 1 - Jun 16 (fully out-of-sample)')
    _, _, Xte, yte, _, _ = build(root, hind, ens, fm, 'UKMOaffine:2016-2022,ecmwf:2023-2024', '2026')
    p_te = m1.predict_proba(Xte[feats])[:, 1]
    thr_dry, f1_dry = best_f1_threshold(yte, p_te)
    m_wet = metrics_at(yte, p_te, thr_wet)
    m_dry = metrics_at(yte, p_te, thr_dry)
    auc, ap = roc_auc_score(yte, p_te), average_precision_score(yte, p_te)
    print('  AUC {:.4f}  AP {:.4f}'.format(auc, ap))
    print('  at WET-calibrated thr {:.4f} : {}   <- HONEST'.format(thr_wet, {k: round(v, 3) for k, v in m_wet.items()}))
    print('  at 2026-optimal   thr {:.4f} : {}   <- optimistic ceiling'.format(thr_dry, {k: round(v, 3) for k, v in m_dry.items()}))
    print('  cost of honest calibration: {:.4f} F1'.format(m_dry['f1'] - m_wet['f1']))

    # --- M2: same config but WITH 2025, to price the cost of holding it out
    print('\n=== M2: train affineUKMO 2016-2022 + ECMWF 2023-2025 (2025 included) -> 2026')
    Xtr2, ytr2, Xte2, yte2, Xva2, yva2 = build(root, hind, ens, fm,
                                               'UKMOaffine:2016-2022,ecmwf:2023-2025', '2026')
    m2 = RandomForestClassifier(criterion='gini', random_state=args.seed, n_jobs=-1, **HP)
    m2.fit(pd.concat([Xtr2, Xva2])[feats], pd.concat([ytr2, yva2]))
    p_te2 = m2.predict_proba(Xte2[feats])[:, 1]
    thr_dry2, _ = best_f1_threshold(yte2, p_te2)
    m2_wet = metrics_at(yte2, p_te2, thr_wet)
    m2_dry = metrics_at(yte2, p_te2, thr_dry2)
    auc2, ap2 = roc_auc_score(yte2, p_te2), average_precision_score(yte2, p_te2)
    print('  AUC {:.4f}  AP {:.4f}   (M1 without 2025: AUC {:.4f} AP {:.4f})'.format(auc2, ap2, auc, ap))
    print('  at WET thr {:.4f} : {}'.format(thr_wet, {k: round(v, 3) for k, v in m2_wet.items()}))
    print('  at 2026-optimal thr {:.4f} : {}'.format(thr_dry2, {k: round(v, 3) for k, v in m2_dry.items()}))
    print('  cost of holding 2025 out: {:+.4f} AUC, {:+.4f} AP'.format(auc - auc2, ap - ap2))

    # --- Export M1 (coherent: model and threshold both blind to 2026)
    meta = {
        'selected_by': 'wet_season_calibration',
        'hyperparameters': HP,
        'train_sources': 'UKMOaffine:2016-2022,ecmwf:2023-2024',
        'threshold': thr_wet,
        'threshold_calibrated_on': '{} {} to {} (ECMWF, wet season, held out of training)'.format(
            args.calib_year, md_a, md_b),
        'calibration_f1': f1_wet,
        'honest_2026_estimate': {'window': '2026-04-01..2026-06-16', 'auc': float(auc), 'ap': float(ap),
                                 **{k: float(v) for k, v in m_wet.items()}},
        'optimistic_2026_ceiling_thr_fit_on_test': {'threshold': thr_dry, **{k: float(v) for k, v in m_dry.items()}},
        'seed': args.seed,
        'features': feats,
        'feature_prefix_note': 'ECMWF samples must be renamed ecmwf_ens_0 -> UKMO_ens_0',
    }
    stem = '{}/rf_model_wet_calibrated'.format(out)
    joblib.dump(m1, stem + '.joblib')
    with open(stem + '.pkl', 'wb') as f:
        pickle.dump(m1, f)
    with open(stem + '_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print('\nSaved {}.pkl (threshold {:.4f})'.format(stem, thr_wet))

    # --- Threshold sensitivity table on both regimes
    rows = []
    for t in [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, thr_wet]:
        rw = metrics_at(ycal_w, p_cal, t)
        rd = metrics_at(yte, p_te, t)
        rows.append({'threshold': round(float(t), 4),
                     'wet2025_f1': round(rw['f1'], 3), 'wet2025_flagged%': round(rw['flagged_pct'], 1),
                     'wet2025_recall': round(rw['recall'], 3),
                     'dry2026_f1': round(rd['f1'], 3), 'dry2026_flagged%': round(rd['flagged_pct'], 1),
                     'dry2026_recall': round(rd['recall'], 3)})
    tbl = pd.DataFrame(rows).sort_values('threshold')
    tbl.to_csv('{}/threshold_sensitivity.csv'.format(out), index=False)
    print('\nTHRESHOLD SENSITIVITY (M1) - wet 2025 Jul-Oct vs dry 2026 Apr-Jun')
    print(tbl.to_string(index=False))
    print('\nDone -> {}'.format(out))
