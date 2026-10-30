"""
Publication-quality Precision-Recall curves for the varying precipitation
forecast experiment.

Generates a single figure with three side-by-side panels (a, b, c) showing the
PR curves for the Random Forest (RF), Gradient Boosting (GB) and eXtreme
Gradient Boosting (XGB) models. Each panel overlays the ensemble members of the
three precipitation forecast providers (KMA, UKMO, NCEP), coloured by provider.

The three input CSVs come from Weights & Biases exports and use two slightly
different schemas, which are both handled here:
  - RF:        columns 'name', 'precision', 'recall'   (run names like 'KMA-Ens0-...')
  - GB / XGB:  columns 'Name', 'Precision', 'Recall'   (run names like 'KMA_0')
In both cases the provider is identified from the substring KMA / UKMO / NCEP.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_DIR = os.environ.get('LANDSLIDE_PROJECT_DIR', 'data')  # see README > Data

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_DIR = PROJECT_DIR + "/Nature_Comms"

# (file, panel label, panel title) for each model, in display order.
MODELS = [
    ("RF_Forecast_Varying_All.csv", "a", "Random Forest"),
    ("GB_Forecast_Varying_All.csv", "b", "Gradient Boosting"),
    ("XGB_Forecast_Varying_All.csv", "c", "eXtreme Gradient Boosting"),
]

# Forecast providers and their styling. Colours are from the Okabe-Ito
# colourblind-safe palette; a distinct linestyle per provider adds redundancy
# so the panels remain legible in greyscale / print.
PROVIDERS = [
    ("KMA",  "#0072B2", "-"),    # blue
    ("UKMO", "#D55E00", "--"),   # vermillion
    ("NCEP", "#009E73", ":"),    # bluish green
]

# Plot style:
#   "overlay" -> every ensemble member drawn as a thin translucent line.
#   "band"    -> per-provider mean curve with a shaded ensemble spread.
STYLE = "band"

OUT_BASENAME = os.path.join(
    DATA_DIR,
    "3panel_varying_band" if STYLE == "band" else "3panel_varying_publication",
)

# Common recall grid onto which each run is interpolated before averaging
# (used only for the "band" style).
RECALL_GRID = np.linspace(0.0, 1.0, 201)

# --------------------------------------------------------------------------- #
# Plot styling
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 0.8,
    "savefig.dpi": 400,
})


def provider_of(name):
    """Return the forecast provider for a given run name, or None."""
    for provider, _, _ in PROVIDERS:
        if provider in name:
            return provider
    return None


def load(csv_path):
    """Load a PR CSV and normalise its column names to Name/Precision/Recall."""
    df = pd.read_csv(csv_path)
    rename = {}
    for col in df.columns:
        low = col.lower()
        if low == "name":
            rename[col] = "Name"
        elif low == "precision":
            rename[col] = "Precision"
        elif low == "recall":
            rename[col] = "Recall"
    return df.rename(columns=rename)


def interp_precision(sub):
    """Interpolate one run's precision onto the common recall grid.

    PR curves are monotonic-ish but unordered in the export, so sort by recall
    first. np.interp clamps outside the data range, which is what we want at the
    grid extremes.
    """
    sub = sub.sort_values("Recall")
    return np.interp(RECALL_GRID, sub["Recall"].values, sub["Precision"].values)


fig, axes = plt.subplots(
    1, 3, figsize=(15, 3.8), sharey=True, constrained_layout=True
)

provider_colors = {p: c for p, c, _ in PROVIDERS}
provider_styles = {p: s for p, _, s in PROVIDERS}

for ax, (fname, label, title) in zip(axes, MODELS):
    df = load(os.path.join(DATA_DIR, fname))

    if STYLE == "band":
        # Group runs by provider, interpolate each onto the common recall grid,
        # then draw the mean curve with a shaded min-max ensemble spread.
        for provider, color, style in PROVIDERS:
            runs = [n for n in df["Name"].unique() if provider in n]
            if not runs:
                continue
            curves = np.vstack([
                interp_precision(df[df["Name"] == n]) for n in runs
            ])
            mean = curves.mean(axis=0)
            std = curves.std(axis=0)
            # Shade mean +/- 1 standard deviation, clipped to the [0, 1] axes.
            lo = np.clip(mean - std, 0.0, 1.0)
            hi = np.clip(mean + std, 0.0, 1.0)
            ax.fill_between(RECALL_GRID, lo, hi, color=color, alpha=0.18, linewidth=0)
            ax.plot(
                RECALL_GRID, mean,
                color=color, linestyle=style, linewidth=2.2,
            )
    else:
        # Overlay every ensemble member as a thin translucent line.
        for run_name in df["Name"].unique():
            provider = provider_of(run_name)
            if provider is None:
                continue
            sub = df[df["Name"] == run_name].sort_values("Recall")
            ax.plot(
                sub["Recall"], sub["Precision"],
                color=provider_colors[provider],
                linestyle=provider_styles[provider],
                linewidth=1.3,
                alpha=0.7,
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_title(title, pad=8)
    ax.grid(True, linestyle="-", linewidth=0.4, alpha=0.25)
    ax.set_axisbelow(True)

    # Panel label (a, b, c) in the top-left corner, outside the data.
    ax.text(
        -0.02, 1.04, label, transform=ax.transAxes,
        fontsize=16, fontweight="bold", va="bottom", ha="right",
    )

axes[0].set_ylabel("Precision")

# Single legend (one entry per provider) on the first panel only, in the empty
# lower-left region so it does not overlap the curves.
legend_handles = [
    Line2D([0], [0], color=c, linestyle=s, linewidth=2.2, label=p)
    for p, c, s in PROVIDERS
]
axes[0].legend(
    handles=legend_handles,
    title="Dynamical forecast product",
    loc="lower left",
    frameon=True,
    framealpha=0.9,
    edgecolor="0.7",
    fontsize=11,
    title_fontsize=12,
)

fig.savefig(f"{OUT_BASENAME}.png", bbox_inches="tight")
fig.savefig(f"{OUT_BASENAME}.pdf", bbox_inches="tight")
print(f"Saved:\n  {OUT_BASENAME}.png\n  {OUT_BASENAME}.pdf")
plt.show()
