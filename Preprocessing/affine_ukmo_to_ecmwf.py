"""
Affine bias-correction of the UKMO forecast features onto the ECMWF scale.

The classifier is trained on historical UKMO forecasts but deployed/tested on
ECMWF (the operational feed, and the only source with daily data covering
Apr-Jun 2026). A model trained on one source and fed the other transfers poorly.
This maps each UKMO forecast feature onto ECMWF's marginal distribution:

    x_affine = a * x_ukmo + b,     a = sd_ecmwf / sd_ukmo
                                   b = mean_ecmwf - a * mean_ukmo

i.e. moment matching, fitted per feature column on the years where BOTH sources
have daily data (2023-2025, monsoon window), paired on (date, district).

What this CANNOT fix: the two sources correlate only ~0.6-0.85 on paired
district-days, so they carry genuinely different information. No affine map
removes that scatter - it only aligns location and spread.

Writes:  LabelledData_{hindcast}/UKMOaffine/ensemble_0/{year}_windowsize14_district.csv
         with forecast columns renamed UKMO_ens_0_* -> UKMOaffine_ens_0_*
         (run_classifier.py's --train_sources renames them back to the training
         feature names).
         Plus affine_coefficients.csv recording every fitted (a, b).
"""

import os
import argparse
import numpy as np
import pandas as pd


def get_args():
    p = argparse.ArgumentParser(description='Affine bias-correct UKMO forecasts onto the ECMWF scale')
    p.add_argument('--root_dir', required=True, help='Root directory of data')
    p.add_argument('--hindcast', default='GPMv07', help='Hindcast source (dir name)')
    p.add_argument('--fit_years', default='2023,2024,2025',
                   help='Years with daily data in BOTH sources, used to fit the transform')
    p.add_argument('--apply_years', default='2016,2017,2018,2019,2020,2021,2022,2023,2024,2025',
                   help='UKMO years to transform and write out')
    p.add_argument('--ens', default='0', help='Ensemble member')
    return p.parse_args()


def forecast_cols(df, prefix):
    return [c for c in df.columns if c.startswith(prefix)]


def fit_affine(root, hind, ens, fit_years):
    """Per-feature (a, b) matching UKMO's mean/sd to ECMWF's, on paired monsoon district-days."""
    u_all, e_all = [], []
    for y in fit_years:
        u = pd.read_csv('{}/LabelledData_{}/UKMO/ensemble_{}/{}_windowsize14_district.csv'.format(root, hind, ens, y))
        e = pd.read_csv('{}/LabelledData_{}/ecmwf/ensemble_{}/{}_windowsize14_district.csv'.format(root, hind, ens, y))
        u = u.rename(columns=lambda c: c.replace('UKMO_ens_{}'.format(ens), 'F'))
        e = e.rename(columns=lambda c: c.replace('ecmwf_ens_{}'.format(ens), 'F'))
        keep = ['date', 'district'] + forecast_cols(u, 'F')
        m = u[keep].merge(e[keep], on=['date', 'district'], suffixes=('_u', '_e'))
        m = m[(m.date >= '{}-04-01'.format(y)) & (m.date <= '{}-10-31'.format(y))]
        u_all.append(m[[c + '_u' for c in forecast_cols(u, 'F')]].rename(columns=lambda c: c[:-2]))
        e_all.append(m[[c + '_e' for c in forecast_cols(e, 'F')]].rename(columns=lambda c: c[:-2]))
        print('  fit year {}: {} paired district-days'.format(y, len(m)))

    U = pd.concat(u_all, ignore_index=True)
    E = pd.concat(e_all, ignore_index=True)

    rows = []
    for c in U.columns:
        su, se = U[c].std(), E[c].std()
        a = se / su if su > 0 else 1.0
        b = E[c].mean() - a * U[c].mean()
        rows.append({'feature': c.replace('F', 'UKMO_ens_{}'.format(ens)), 'a': a, 'b': b,
                     'ukmo_mean': U[c].mean(), 'ukmo_sd': su, 'ecmwf_mean': E[c].mean(), 'ecmwf_sd': se})
    coef = pd.DataFrame(rows).set_index('feature')
    return coef


if __name__ == '__main__':
    args = get_args()
    root, hind, ens = args.root_dir, args.hindcast, args.ens
    fit_years = args.fit_years.split(',')
    apply_years = args.apply_years.split(',')

    print('Fitting affine UKMO -> ECMWF on {}'.format(fit_years))
    coef = fit_affine(root, hind, ens, fit_years)
    print('\nFitted coefficients (x_affine = a*x + b):')
    print(coef[['a', 'b', 'ukmo_mean', 'ecmwf_mean']].round(3).to_string())

    out_dir = '{}/LabelledData_{}/UKMOaffine/ensemble_{}'.format(root, hind, ens)
    os.makedirs(out_dir, exist_ok=True)
    coef.to_csv('{}/affine_coefficients.csv'.format(out_dir))

    src_prefix = 'UKMO_ens_{}'.format(ens)
    dst_prefix = 'UKMOaffine_ens_{}'.format(ens)
    print('\nApplying to UKMO years {}'.format(apply_years))
    for y in apply_years:
        path = '{}/LabelledData_{}/UKMO/ensemble_{}/{}_windowsize14_district.csv'.format(root, hind, ens, y)
        if not os.path.exists(path):
            print('  {} missing, skipping'.format(path))
            continue
        df = pd.read_csv(path)
        n_neg = 0
        n_tot = 0
        for c in forecast_cols(df, src_prefix):
            a, b = coef.loc[c, 'a'], coef.loc[c, 'b']
            v = a * df[c].values + b
            n_neg += int((v < 0).sum())
            n_tot += len(v)
            # Precipitation cannot be negative; the intercept can push small values
            # below zero, so clip. Affects the low tail only.
            df[c] = np.clip(v, 0, None)
        df = df.rename(columns=lambda c: c.replace(src_prefix, dst_prefix))
        df.to_csv('{}/{}_windowsize14_district.csv'.format(out_dir, y), index=False)
        print('  {}: {} rows written ({:.2f}% of forecast values clipped at 0)'.format(
            y, len(df), 100 * n_neg / n_tot))

    print('\nDone -> {}'.format(out_dir))
