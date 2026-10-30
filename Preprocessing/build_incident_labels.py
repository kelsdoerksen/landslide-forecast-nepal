"""
Build a single combined, de-duplicated landslide incident file (2016-2026) from
the multiple Bipad portal downloads, for use as the label source in
generate_samples_hindcast_forecast.py.

- Parses 'Incident on' to ISO YYYY-MM-DD (files use either ISO or m/d/y).
- Normalizes Bipad district spellings to the naming used in the precip CSVs.
- Keeps only Hazard == 'Landslide'.
- De-duplicates records that appear in more than one download.
"""

import os
import pandas as pd

PROJECT_DIR = os.environ.get('LANDSLIDE_PROJECT_DIR', 'data')  # see README > Data

root_dir = PROJECT_DIR + '/Monsoon2024_Prep'

SOURCE_FILES = [
    'bipad_records_2016-2024_july172025download.csv',        # 2016-03 -> 2025-01 (ISO)
    'Bipad_Incidents_2023-2024_May82025Download.csv',        # 2023-01 -> 2024-12 (m/d/y)
    'bipad_incidents_April12025-July12026_downloadJuly72026.csv',  # 2025-04 -> 2026-06 (ISO)
    'incidents-bipad-April12026-June302026.csv',              # 2026-03 -> 2026-06 (ISO), fresher pull
    # Full 2026 monsoon pull downloaded 2026-08-28. Listed last so it wins the
    # de-dupe; this is the first source with July/August 2026.
    '2026_Monsoon_Testing_Operations/2026-incidents-April-Aug-Aug282026-download.csv',
]

OUTPUT_FILE = 'combined_landslide_incidents_2016-2026.csv'

# Bipad district spellings -> naming used in `districts` / precip CSVs
district_name_map = {
    'Chitwan': 'Chitawan', 'Kavrepalanchok': 'Kabhrepalanchok', 'Makwanpur': 'Makawanpur',
    'Nawalparasi East': 'Nawalparasi_E', 'Nawalparasi West': 'Nawalparasi_W',
    'Rukum East': 'Rukum_E', 'Rukum West': 'Rukum_W',
}

# Fields that identify a unique incident across downloads
DEDUP_KEYS = ['Incident on', 'District', 'Municipality', 'Ward', 'Title']


def parse_dates_iso(series):
    """Parse a mixed-format date column to ISO YYYY-MM-DD strings."""
    s = series.astype(str).str.strip()
    iso = pd.to_datetime(s, format='%Y-%m-%d', errors='coerce')
    us = pd.to_datetime(s, format='%m/%d/%y', errors='coerce')
    dmy = pd.to_datetime(s, format='%d/%m/%Y', errors='coerce')
    parsed = iso.fillna(us).fillna(dmy)
    return parsed.dt.strftime('%Y-%m-%d')


if __name__ == "__main__":
    frames = []
    for f in SOURCE_FILES:
        path = '{}/{}'.format(root_dir, f)
        df = pd.read_csv(path)
        df['Hazard'] = df['Hazard'].astype(str).str.strip()
        df = df[df['Hazard'] == 'Landslide'].copy()
        df['Incident on'] = parse_dates_iso(df['Incident on'])
        df['District'] = df['District'].replace(district_name_map)
        bad = df['Incident on'].isna().sum()
        print('{}: {} landslide rows ({} unparseable dates dropped)'.format(f, len(df), bad))
        df = df.dropna(subset=['Incident on'])
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    n_before = len(combined)

    # Normalize dedup-key blanks so identical incidents match across downloads
    key_norm = combined[DEDUP_KEYS].fillna('').astype(str)
    combined = combined.loc[~key_norm.duplicated(keep='first')].reset_index(drop=True)
    print('\nCombined: {} rows -> {} after de-dup ({} duplicates removed)'.format(
        n_before, len(combined), n_before - len(combined)))

    combined = combined.sort_values('Incident on').reset_index(drop=True)
    out_path = '{}/{}'.format(root_dir, OUTPUT_FILE)
    combined.to_csv(out_path, index=False)

    yr = pd.to_datetime(combined['Incident on']).dt.year
    print('Date span: {} -> {}'.format(combined['Incident on'].min(), combined['Incident on'].max()))
    print('Unique (District, date) landslide-days: {}'.format(
        combined[['District', 'Incident on']].drop_duplicates().shape[0]))
    print('Rows per year:')
    print(yr.value_counts().sort_index().to_string())
    print('\nWrote {}'.format(out_path))
