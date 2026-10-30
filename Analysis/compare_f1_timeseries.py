"""
Overlay the 14-day rolling F1 timeseries of several run folders on a shared
day-of-year axis, so seasons can be compared directly (e.g. the full 2025
monsoon vs the truncated 2026 Apr-Jun window).

Reads each run's f1_timeseries.csv (written by Analysis/analyze_run.py).

Usage:
  python Analysis/compare_f1_timeseries.py --out <PNG> \
      --run 2025:<dir>/analysis_2025_thr0.1720 --run 2026:<dir>/analysis_2026_thr0.1720
"""

import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = ['slateblue', 'firebrick', 'seagreen', 'darkorange']


def get_args():
    p = argparse.ArgumentParser(description='Overlay rolling-F1 timeseries across seasons')
    p.add_argument('--run', action='append', required=True, help='LABEL:/path/to/run_dir (repeatable)')
    p.add_argument('--out', required=True)
    p.add_argument('--threshold', default=None, help='Shown in the title only')
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    for i, spec in enumerate(args.run):
        label, run_dir = spec.split(':', 1)
        ts = pd.read_csv('{}/f1_timeseries.csv'.format(run_dir))
        ts['date'] = pd.to_datetime(ts['date'])
        ts['doy'] = ts['date'].dt.dayofyear
        c = COLORS[i % len(COLORS)]
        ax2.bar(ts['doy'], ts['landslide_count'], color=c, alpha=0.15, width=1.0)
        ax1.plot(ts['doy'], ts['f1'], color=c, lw=2.2, label=label)

    # Month gridlines on the day-of-year axis (non-leap reference).
    months = {'Apr': 91, 'May': 121, 'Jun': 152, 'Jul': 182,
              'Aug': 213, 'Sep': 244, 'Oct': 274, 'Nov': 305}
    for d in months.values():
        ax1.axvline(d, color='0.85', lw=0.8, zorder=0)
    ax1.set_xticks(list(months.values()))
    ax1.set_xticklabels(list(months.keys()))

    ax1.set_ylim(0, 1)
    ax1.set_ylabel('14-day rolling F1')
    ax2.set_ylabel('Landsliding district count (shaded)')
    ax1.set_xlabel('Day of year')
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)
    ax1.legend(loc='upper left')
    title = 'Rolling F1 by season'
    if args.threshold:
        title += ' (threshold {})'.format(args.threshold)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print('Wrote {}'.format(args.out))
