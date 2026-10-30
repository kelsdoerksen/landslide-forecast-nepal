"""
Single-run analysis plots for a classifier result folder.

Produces, saved INTO the run directory:
  - fi_bar.png            RF Gini feature importance, coloured by feature group
  - fi_group_pie.png      importance grouped into forecast / observed precip / geomorphic
  - pr_curve.png          precision-recall curve with best-F1 point and no-skill baseline
  - f1_timeseries.png     daily F1 over the monsoon with landslide-count bars behind it
  - cm_counts_map.png     per-district TP/FP/FN/TN count maps          (needs geopandas)
  - cm_rates_map.png      per-district TP/FP/FN/TN rate maps           (needs geopandas)
  - cm_f1_map.png         per-district F1 map                          (needs geopandas)
  - confusion_matrix.geojson

This is a headless, single-run version of the Analysis/ paper scripts
(feature_importance.py, pr_curves.py, confusion_matrix.py, f1_aggregation.py),
which otherwise aggregate over 5 seeded runs per year and read paper-specific
inputs. Threshold defaults to the best-F1 point of the run's own PR curve.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score

PROJECT_DIR = os.environ.get('LANDSLIDE_PROJECT_DIR', 'data')  # see README > Data


def get_args():
    p = argparse.ArgumentParser(description='Single-run analysis plots')
    p.add_argument('--run', default=None, help='Run folder name under Results/GPMv07')
    p.add_argument('--root_dir', required=True, help='Root directory of data')
    p.add_argument('--run_dir', default=None,
                   help='Full path to the run folder. Overrides --run; use for runs saved outside Results/GPMv07')
    p.add_argument('--test_year', required=True, help='Test year (e.g. 2023)')
    p.add_argument('--threshold', default=None, help='Decision threshold; default = best-F1 of the PR curve')
    p.add_argument('--geojson', default=None, help='District boundaries geojson (default Nature_Comms path)')
    return p.parse_args()


def feature_color(name):
    if 'ens' in name or 'UKMO' in name or 'ECMWF' in name or 'ecmwf' in name:
        return 'slateblue'      # precipitation forecast
    if 'tminus' in name or 'GPM' in name or 'GSMaP' in name:
        return 'deepskyblue'    # precipitation observation
    return 'mediumseagreen'     # geomorphic (dem/slope/aspect/lc)


def best_f1_threshold(pr_df):
    p = pr_df['precision'].values
    r = pr_df['recall'].values
    t = pr_df['thresholds'].values
    with np.errstate(divide='ignore', invalid='ignore'):
        f = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
    ix = int(np.nanargmax(f))
    return float(t[ix]) if not np.isnan(t[ix]) else 0.5, float(f[ix])


def plot_feature_importance(run_dir):
    fi = pd.read_csv('{}/FI.csv'.format(run_dir)).drop(columns=['Unnamed: 0'], errors='ignore')
    fi_dict = fi.iloc[0].to_dict()
    items = sorted(fi_dict.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = [feature_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.bar(names, vals, width=0.6, color=colors)
    plt.xticks(rotation=90)
    plt.ylabel('Gini Feature Importance')
    plt.title('RF Feature Importance')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ['slateblue', 'deepskyblue', 'mediumseagreen']]
    ax.legend(handles, ['Precip Forecast', 'Precip Observation', 'Geomorphic'])
    plt.tight_layout()
    plt.savefig('{}/fi_bar.png'.format(run_dir), dpi=200)
    plt.close()

    forecast_val = sum(v for k, v in fi_dict.items() if feature_color(k) == 'slateblue')
    obs_val = sum(v for k, v in fi_dict.items() if feature_color(k) == 'deepskyblue')
    geo_val = sum(v for k, v in fi_dict.items() if feature_color(k) == 'mediumseagreen')
    fig, ax = plt.subplots()
    ax.pie([forecast_val, obs_val, geo_val],
           labels=['Precip Forecast', 'Precip Observation', 'Geomorphic'],
           colors=['slateblue', 'deepskyblue', 'mediumseagreen'],
           autopct='%1.1f%%', textprops={'color': 'w'})
    ax.set_title('Feature Importance by Group')
    plt.savefig('{}/fi_group_pie.png'.format(run_dir), dpi=200, bbox_inches='tight')
    plt.close()


def plot_pr_curve(run_dir, preds, thr, f1_at_thr):
    pr = pd.read_csv('{}/precision_recall.csv'.format(run_dir))
    baseline = (preds['groundtruth'] == 1).mean()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(pr['recall'], pr['precision'], marker='.', ms=3, label='RF')
    # best-F1 point
    p = pr['precision'].values; r = pr['recall'].values; t = pr['thresholds'].values
    ix = int(np.nanargmin(np.abs(t - thr)))
    ax.scatter(r[ix], p[ix], color='black', zorder=5,
               label='Best threshold {:.3f} (F1={:.3f})'.format(thr, f1_at_thr))
    ax.axhline(baseline, color='r', ls='--', label='No skill ({:.3f})'.format(baseline))
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('{}/pr_curve.png'.format(run_dir), dpi=200)
    plt.close()


def plot_f1_timeseries(run_dir, preds, year, thr, window=14):
    """
    F1 timeseries over the monsoon. Daily F1 is degenerate when positives are
    sparse (early season often has <=1 landsliding district/day, so daily F1
    flat-lines at 0). We therefore report a `window`-day ROLLING pooled F1
    (matches the 14-day forecast horizon) as the primary 'f1' series, and keep
    the raw daily F1 as 'f1_daily'.
    """
    df = preds.copy()
    df['pred'] = (df['model soft predictions'] >= thr).astype(int)
    rows = []
    for d, g in df.groupby('date'):
        tp = int(((g['groundtruth'] == 1) & (g['pred'] == 1)).sum())
        pos = int((g['groundtruth'] == 1).sum())
        predpos = int((g['pred'] == 1).sum())
        rows.append({'date': d, 'tp': tp, 'fp': predpos - tp, 'fn': pos - tp,
                     'f1_daily': f1_score(g['groundtruth'], g['pred'], zero_division=0),
                     'landslide_count': pos})
    ts = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
    roll = ts[['tp', 'fp', 'fn']].rolling(window, min_periods=max(1, window // 2)).sum()
    denom = (2 * roll['tp'] + roll['fp'] + roll['fn'])
    ts['f1'] = np.where(denom > 0, 2 * roll['tp'] / denom, 0.0)
    dates = ts['date'].tolist()

    def idx(m):
        tag = '{}-{}-01'.format(year, m)
        return dates.index(tag) if tag in dates else None
    ticks = [i for i in (idx('04'), idx('05'), idx('06'), idx('07'), idx('08'), idx('09'), idx('10')) if i is not None]

    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax2 = ax1.twinx()
    ax2.bar(range(len(ts)), ts['landslide_count'], color='gray', alpha=0.3, width=1.0,
            label='Landsliding districts')
    ax1.plot(range(len(ts)), ts['f1_daily'], color='lightsteelblue', lw=0.9, label='Daily F1 (raw)')
    ax1.plot(range(len(ts)), ts['f1'], color='slateblue', lw=2.0, label='14-day rolling F1')
    ax1.set_ylim(0, 1); ax1.set_ylabel('F1 Score'); ax2.set_ylabel('Landsliding district count')
    ax1.set_xticks(ticks); ax1.set_xticklabels([dates[i][:7] for i in ticks])
    ax1.set_xlabel('Date')
    ax1.set_zorder(ax2.get_zorder() + 1); ax1.patch.set_visible(False)
    lines, labels = ax1.get_legend_handles_labels()
    bars, blabels = ax2.get_legend_handles_labels()
    ax1.legend(lines + bars, labels + blabels, loc='upper left')
    plt.title('F1 over {} monsoon (14-day rolling, threshold {:.3f})'.format(year, thr))
    plt.tight_layout()
    plt.savefig('{}/f1_timeseries.png'.format(run_dir), dpi=200)
    plt.close()
    ts.to_csv('{}/f1_timeseries.csv'.format(run_dir), index=False)
    # Aggregator-compatible per-run file consumed by aggregating_f1_timeseries.py (expects doy, f1)
    ts.rename(columns={'date': 'doy'}).to_csv(
        '{}/f1_timeseries_{}_thr{}.csv'.format(run_dir, year, thr), index=False)


def plot_cm_maps(run_dir, preds, geojson_path, thr):
    import geopandas as gpd
    gdf = gpd.read_file(geojson_path)
    # Normalize district boundary names to the sample naming
    repl = {'Rukum East': 'Rukum_E', 'Rukum West': 'Rukum_W',
            'Parasi': 'Nawalparasi_W', 'Nawalpur': 'Nawalparasi_E',
            'Nawalparasi East': 'Nawalparasi_E', 'Nawalparasi West': 'Nawalparasi_W'}
    gdf['DISTRICT'] = gdf['DISTRICT'].replace(repl)

    recs = []
    for dist, g in preds.groupby('district'):
        labels = g['groundtruth'].values
        pred = (g['model soft predictions'].values >= thr).astype(int)
        cm = confusion_matrix(labels, pred, labels=[0, 1])
        TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        tpr = TP / (TP + FN) if (TP + FN) else 0.0
        fpr = FP / (FP + TN) if (FP + TN) else 0.0
        fnr = FN / (FN + TP) if (TP + FN) else 0.0
        tnr = TN / (TN + FP) if (FP + TN) else 0.0
        f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) else 0.0
        recs.append({'DISTRICT': dist, 'tp': TP, 'fp': FP, 'fn': FN, 'tn': TN,
                     'tp_rate': tpr, 'fp_rate': fpr, 'fn_rate': fnr, 'tn_rate': tnr,
                     'f1': f1, 'Landslide Count': int(labels.sum())})
    metrics = pd.DataFrame(recs)
    combined = gdf.merge(metrics, on='DISTRICT', how='left').fillna(0)
    combined.to_file('{}/confusion_matrix.geojson'.format(run_dir), driver='GeoJSON')

    # Counts (2x2)
    mx = {c: max(combined[c].max(), 1) for c in ['tp', 'fp', 'fn', 'tn']}
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    for (col, cmap, title, a) in [('tp', 'Reds', 'True Positives', ax[0, 0]),
                                  ('fp', 'Blues', 'False Positives', ax[0, 1]),
                                  ('fn', 'Greens', 'False Negatives', ax[1, 0]),
                                  ('tn', 'Oranges', 'True Negatives', ax[1, 1])]:
        pl = combined.plot(ax=a, column=col, edgecolor='black', linewidth=0.4, cmap=cmap, vmin=0, vmax=mx[col])
        a.set_title(title); a.axis('off')
        fig.colorbar(pl.collections[0], ax=a, label='Count')
    plt.tight_layout(); plt.savefig('{}/cm_counts_map.png'.format(run_dir), dpi=200); plt.close()

    # Rates (2x2)
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    for (col, cmap, title, a) in [('tp_rate', 'Reds', 'True Positive Rate', ax[0, 0]),
                                  ('fp_rate', 'Blues', 'False Positive Rate', ax[0, 1]),
                                  ('fn_rate', 'Greens', 'False Negative Rate', ax[1, 0]),
                                  ('tn_rate', 'Oranges', 'True Negative Rate', ax[1, 1])]:
        pl = combined.plot(ax=a, column=col, edgecolor='black', linewidth=0.4, cmap=cmap, vmin=0, vmax=1)
        a.set_title(title); a.axis('off')
        fig.colorbar(pl.collections[0], ax=a, label='Rate')
    plt.tight_layout(); plt.savefig('{}/cm_rates_map.png'.format(run_dir), dpi=200); plt.close()

    # F1
    fig, ax = plt.subplots(figsize=(9, 7))
    pl = combined.plot(ax=ax, column='f1', edgecolor='black', linewidth=0.4, cmap='Reds', vmin=0, vmax=1)
    ax.set_title('Per-district F1'); ax.axis('off')
    fig.colorbar(pl.collections[0], ax=ax, label='F1 Score')
    plt.tight_layout(); plt.savefig('{}/cm_f1_map.png'.format(run_dir), dpi=200); plt.close()


if __name__ == '__main__':
    args = get_args()
    if not args.run_dir and not args.run:
        raise SystemExit('Pass either --run (a folder under Results/GPMv07) or --run_dir (a full path)')
    run_dir = args.run_dir or '{}/Results/GPMv07/{}'.format(args.root_dir, args.run)
    year = args.test_year
    geojson = args.geojson or (PROJECT_DIR + '/'
                               'Nature_Comms/Nepal_District_Boundaries.geojson')

    preds = pd.read_csv('{}/predictions_and_groundtruth.csv'.format(run_dir))
    pr_df = pd.read_csv('{}/precision_recall.csv'.format(run_dir))
    thr, f1_at = best_f1_threshold(pr_df) if args.threshold is None else (float(args.threshold), np.nan)
    print('Run {} | year {} | threshold {:.4f}'.format(args.run, year, thr))

    if os.path.exists('{}/FI.csv'.format(run_dir)):
        plot_feature_importance(run_dir); print('  fi_bar.png, fi_group_pie.png')
    else:
        print('  [skipped feature importance: no FI.csv in run folder]')
    plot_pr_curve(run_dir, preds, thr, f1_at); print('  pr_curve.png')
    plot_f1_timeseries(run_dir, preds, year, thr); print('  f1_timeseries.png')
    try:
        plot_cm_maps(run_dir, preds, geojson, thr); print('  cm_counts_map.png, cm_rates_map.png, cm_f1_map.png')
    except ImportError:
        print('  [skipped CM maps: geopandas not available]')
    print('Done -> {}'.format(run_dir))
