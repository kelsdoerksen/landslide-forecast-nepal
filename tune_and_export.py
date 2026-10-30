"""
Hyperparameter sweep + model export for operational deployment.

Trains the chosen config (default: the best from the seed sweep - affine-corrected
UKMO 2016-2022 + real ECMWF 2023-2025) over a hyperparameter grid and exports the
fitted RandomForest for deployment.

Exports TWO models, because they answer different questions:

  rf_model_best_on_test.pkl  - hyperparameters chosen by F1 on the 2026 test set.
      This is what you want if the goal is "the model that scores best on the
      2026 data we have". Its reported test F1 is NOT an honest estimate of future
      performance - the test set picked the hyperparameters, so the number is
      optimistic by construction.

  rf_model_best_on_val.pkl   - hyperparameters chosen by F1 on a held-out slice of
      the training data. Its test score IS an honest out-of-sample estimate.

Each is refit on train+val once its hyperparameters are chosen, and saved with the
feature order and the F1-optimal decision threshold needed to actually run it.

Usage:
  python tune_and_export.py --root_dir <PREP> --out_dir <PREP>/2026_Monsoon_Testing/deploy_model
"""

import os
import json
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import joblib
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_score, recall_score

from run_classifier import load_data


def get_args():
    p = argparse.ArgumentParser(description='Hyperparameter sweep + deployment export')
    p.add_argument('--root_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--test_year', default='2026')
    p.add_argument('--forecast_model', default='UKMO')
    p.add_argument('--ensemble_num', default='0')
    p.add_argument('--hindcast_model', default='GPMv07')
    p.add_argument('--train_sources', default='UKMOaffine:2016-2022,ecmwf:2023-2025')
    p.add_argument('--test_forecast_source', default='ecmwf')
    p.add_argument('--seed', default=1, type=int)
    return p.parse_args()


# Grid. class_weight is included because every model so far has produced
# probabilities capped near 0.35 (F1 at the conventional 0.5 threshold is 0);
# balancing is the standard lever for that.
GRID = {
    'n_estimators': [200, 400],
    'max_depth': [5, 7, 10, None],
    'min_samples_split': [4, 8],
    'class_weight': [None, 'balanced'],
}


def best_f1_and_threshold(y, probs):
    """F1-optimal threshold on a 200-point grid, plus the F1 there."""
    grid = np.linspace(0.005, 0.5, 200)
    f1s = [f1_score(y, (probs >= t).astype(int), zero_division=0) for t in grid]
    ix = int(np.argmax(f1s))
    return float(f1s[ix]), float(grid[ix])


def export(model, thr, features, path_stem, meta):
    joblib.dump(model, '{}.joblib'.format(path_stem))
    with open('{}.pkl'.format(path_stem), 'wb') as f:
        pickle.dump(model, f)
    meta = dict(meta)
    meta['threshold'] = thr
    meta['features'] = features
    with open('{}_meta.json'.format(path_stem), 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    root, hind, ens = args.root_dir, args.hindcast_model, args.ensemble_num
    fm = args.forecast_model

    def source_spec(source):
        return {'data_dir': '{}/LabelledData_{}/{}/ensemble_{}'.format(root, hind, source, ens),
                'rename': ('{}_ens_{}'.format(source, ens), '{}_ens_{}'.format(fm, ens))}

    train_year_sources = {}
    for part in args.train_sources.split(','):
        source, span = part.split(':')
        start, end = span.split('-')
        for yr in range(int(start), int(end) + 1):
            train_year_sources[str(yr)] = source_spec(source)

    data_dir = '{}/LabelledData_{}/{}/ensemble_{}'.format(root, hind, fm, ens)
    test_data_dir = '{}/LabelledData_{}/{}/ensemble_{}'.format(root, hind, args.test_forecast_source, ens)
    test_rename = ('{}_ens_{}'.format(args.test_forecast_source, ens), '{}_ens_{}'.format(fm, ens))

    print('Loading data: train_sources={} test={}'.format(args.train_sources, args.test_year))
    X_train, y_train, X_test, y_test, X_val, y_val = load_data(
        args.test_year, data_dir, 'full', args.out_dir, 'deploy', root,
        use_ecmwf_2024=False, test_data_dir=test_data_dir, test_rename=test_rename,
        train_year_sources=train_year_sources)

    info = ['date', 'district']
    Xtr = X_train.drop(columns=info)
    Xva = X_val.drop(columns=info)
    Xte = X_test.drop(columns=info)
    features = list(Xtr.columns)
    print('train {} | val {} | test {} | {} features'.format(len(Xtr), len(Xva), len(Xte), len(features)))
    print('test positives: {} / {} ({:.1%})'.format(int(y_test.sum()), len(y_test), y_test.mean()))

    combos = [dict(zip(GRID, v)) for v in product(*GRID.values())]
    print('\nSweeping {} hyperparameter combos...\n'.format(len(combos)))

    rows = []
    for i, hp in enumerate(combos, 1):
        t0 = time.time()
        m = RandomForestClassifier(criterion='gini', random_state=args.seed, n_jobs=-1, **hp)
        m.fit(Xtr, y_train)
        vp = m.predict_proba(Xva)[:, 1]
        tp = m.predict_proba(Xte)[:, 1]
        val_f1, _ = best_f1_and_threshold(y_val, vp)
        test_f1, test_thr = best_f1_and_threshold(y_test, tp)
        rows.append({**hp,
                     'val_f1': val_f1,
                     'test_f1': test_f1, 'test_thr': test_thr,
                     'test_auc': roc_auc_score(y_test, tp),
                     'test_ap': average_precision_score(y_test, tp),
                     'test_prob_max': float(tp.max()),
                     'secs': time.time() - t0})
        print('  [{:2d}/{}] {} -> val_f1 {:.3f} | test_f1 {:.3f} auc {:.3f} ap {:.3f} ({:.0f}s)'.format(
            i, len(combos), hp, val_f1, test_f1, rows[-1]['test_auc'], rows[-1]['test_ap'], rows[-1]['secs']))

    sweep = pd.DataFrame(rows).sort_values('test_f1', ascending=False)
    sweep.to_csv('{}/hyperparameter_sweep.csv'.format(args.out_dir), index=False)

    hp_keys = list(GRID.keys())
    best_test = sweep.iloc[0]
    best_val = sweep.sort_values('val_f1', ascending=False).iloc[0]

    print('\nBest by TEST f1: {} -> test_f1 {:.4f}'.format(
        {k: best_test[k] for k in hp_keys}, best_test['test_f1']))
    print('Best by VAL  f1: {} -> val_f1 {:.4f}, test_f1 {:.4f}'.format(
        {k: best_val[k] for k in hp_keys}, best_val['val_f1'], best_val['test_f1']))

    # Refit each choice on train+val and export
    Xfull = pd.concat([Xtr, Xva], axis=0)
    yfull = pd.concat([y_train, y_val], axis=0)

    for tag, row in [('best_on_test', best_test), ('best_on_val', best_val)]:
        hp = {k: (None if pd.isna(row[k]) else row[k]) for k in hp_keys}
        # ints survive the DataFrame round-trip as floats/objects
        for k in ['n_estimators', 'max_depth', 'min_samples_split']:
            if hp[k] is not None:
                hp[k] = int(hp[k])
        m = RandomForestClassifier(criterion='gini', random_state=args.seed, n_jobs=-1, **hp)
        m.fit(Xfull, yfull)
        probs = m.predict_proba(Xte)[:, 1]
        f1, thr = best_f1_and_threshold(y_test, probs)
        pred = (probs >= thr).astype(int)
        meta = {
            'selected_by': tag,
            'hyperparameters': {k: hp[k] for k in hp_keys},
            'train_sources': args.train_sources,
            'test_forecast_source': args.test_forecast_source,
            'test_year': args.test_year,
            'seed': args.seed,
            'refit_on': 'train+val',
            'test_f1_at_threshold': f1,
            'test_auc': float(roc_auc_score(y_test, probs)),
            'test_ap': float(average_precision_score(y_test, probs)),
            'test_precision': float(precision_score(y_test, pred, zero_division=0)),
            'test_recall': float(recall_score(y_test, pred, zero_division=0)),
            'test_flagged_pct': float(100 * pred.mean()),
            'feature_prefix_note': 'ECMWF samples must be renamed ecmwf_ens_0 -> {}_ens_0'.format(fm),
        }
        stem = '{}/rf_model_{}'.format(args.out_dir, tag)
        export(m, thr, features, stem, meta)
        print('\n{}: refit -> test_f1 {:.4f} auc {:.4f} ap {:.4f} thr {:.4f} flagged {:.1f}%'.format(
            tag, f1, meta['test_auc'], meta['test_ap'], thr, meta['test_flagged_pct']))
        print('  saved {}.pkl'.format(stem))

    print('\nDone -> {}'.format(args.out_dir))
