"""
Score the deployed model on a year's ECMWF samples and write a run folder that
Analysis/analyze_run.py can plot.

calibrate_threshold.py fit the deployment threshold on 2025 Jul-Oct but never
persisted the 2025 predictions, so there is no 2025 timeseries to show. This
reloads the shipped model, scores any year/window of ECMWF ensemble-0 samples,
and writes predictions_and_groundtruth.csv / precision_recall.csv / FI.csv in
the same layout run_classifier.py produces.

Usage:
  python score_deployed_model.py --root_dir <PREP> \
      --model <PREP>/2026_Monsoon_Testing/deploy_model/rf_model_wet_calibrated.pkl \
      --year 2025 --out_dir <PREP>/2026_Monsoon_Testing/deploy_model/analysis_2025_thr0.1720
"""

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (precision_recall_curve, f1_score, roc_auc_score,
                             average_precision_score, precision_score, recall_score,
                             confusion_matrix, classification_report)


def get_args():
    p = argparse.ArgumentParser(description='Score the deployed RF on a year of ECMWF samples')
    p.add_argument('--root_dir', required=True)
    p.add_argument('--model', required=True, help='Path to rf_model_*.pkl')
    p.add_argument('--meta', default=None, help='Path to *_meta.json (default: model path with _meta.json)')
    p.add_argument('--year', required=True)
    p.add_argument('--window', default=None, help='MM-DD:MM-DD subset within the year (default: whole file)')
    p.add_argument('--hindcast_model', default='GPMv07')
    p.add_argument('--forecast_source', default='ecmwf')
    p.add_argument('--ens_member', default='0')
    p.add_argument('--threshold', default=None, type=float, help='Default: threshold from meta json')
    p.add_argument('--out_dir', required=True)
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    meta_path = args.meta or args.model.rsplit('.', 1)[0] + '_meta.json'
    meta = json.load(open(meta_path))
    thr = args.threshold if args.threshold is not None else float(meta['threshold'])
    feats = meta['features']
    os.makedirs(args.out_dir, exist_ok=True)

    samples = '{}/LabelledData_{}/{}/ensemble_{}/{}_windowsize14_district.csv'.format(
        args.root_dir, args.hindcast_model, args.forecast_source, args.ens_member, args.year)
    df = pd.read_csv(samples)
    # Forecast columns must carry the training prefix the model was fit on.
    df = df.rename(columns=lambda c: c.replace('{}_ens_{}'.format(args.forecast_source, args.ens_member),
                                               'UKMO_ens_{}'.format(args.ens_member)))
    if args.window:
        a, b = args.window.split(':')
        df = df[(df['date'] >= '{}-{}'.format(args.year, a)) & (df['date'] <= '{}-{}'.format(args.year, b))]
    df = df.sort_values('date').reset_index(drop=True)

    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise SystemExit('Samples are missing {} model features, first few: {}'.format(len(missing), missing[:5]))

    print('Loading {} ...'.format(args.model))
    model = pickle.load(open(args.model, 'rb'))
    probs = model.predict_proba(df[feats])[:, 1]
    y = df['label'].values
    pred = (probs >= thr).astype(int)

    pd.DataFrame({'model soft predictions': probs, 'groundtruth': y,
                  'date': df['date'], 'district': df['district']}).to_csv(
        '{}/predictions_and_groundtruth.csv'.format(args.out_dir), index=False)

    prec, rec, thrs = precision_recall_curve(y, probs)
    pd.DataFrame({'thresholds': np.append(thrs, 1.0), 'precision': prec, 'recall': rec}).to_csv(
        '{}/precision_recall.csv'.format(args.out_dir), index=False)

    pd.DataFrame([model.feature_importances_], index=['importance'], columns=feats).to_csv(
        '{}/FI.csv'.format(args.out_dir))

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    res = {'model': os.path.basename(args.model), 'year': args.year,
           'window': '{} -> {}'.format(df['date'].min(), df['date'].max()),
           'samples': int(len(df)), 'districts': int(df['district'].nunique()),
           'positives': int(y.sum()), 'positive_rate': float(y.mean()),
           'threshold': thr, 'f1': float(f1_score(y, pred, zero_division=0)),
           'precision': float(precision_score(y, pred, zero_division=0)),
           'recall': float(recall_score(y, pred, zero_division=0)),
           'flagged_pct': float(100 * pred.mean()),
           'roc_auc': float(roc_auc_score(y, probs)),
           'average_precision': float(average_precision_score(y, probs)),
           'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)}
    with open('{}/model_testing_results.txt'.format(args.out_dir), 'w') as f:
        json.dump(res, f, indent=2)
    pd.DataFrame(classification_report(y, pred, output_dict=True, zero_division=0)).T.to_csv(
        '{}/Test_classification_report.csv'.format(args.out_dir))

    for k, v in res.items():
        print('  {:18s} {}'.format(k, round(v, 4) if isinstance(v, float) else v))
    print('Wrote -> {}'.format(args.out_dir))
