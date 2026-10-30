"""
Publication-quality Precision-Recall curves for the Random Forest temporal
cross-validation experiment (Nature Comms paper).

For each held-out test year (2016-2024) the model is trained five times; this
script averages the five PR curves for the year and plots all nine yearly mean
curves on a single figure, with a qualitative 9-colour legend keyed to year.

Note on averaging: the per-run PR exports are ordered by decision threshold but
have slightly different lengths (the threshold grid differs run to run). To
avoid fabricating any data via interpolation, the five real curves are averaged
element-wise by threshold rank, truncating to the shortest run so every
averaged point is the mean of five real (recall, precision) points.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

PROJECT_DIR = os.environ.get('LANDSLIDE_PROJECT_DIR', 'data')  # see README > Data

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ROOT_DIR = PROJECT_DIR + "/Monsoon2024_Prep/Results/GPMv07"
SAVE_DIR = PROJECT_DIR + "/Nature_Comms"
OUT_BASENAME = os.path.join(SAVE_DIR, "rf_temporal_cv_pr")
SAVE_YEARLY_CSV = True  # also write the averaged per-year curves to SAVE_DIR

# Five RF runs per held-out test year.
RF_RUNS = {
    2016: ["jumping-sound-1778", "clear-totem-1777", "fragrant-brook-1776", "scarlet-pond-1775", "earnest-fire-1774"],
    2017: ["stilted-cosmos-1783", "likely-voice-1782", "fresh-frost-1781", "denim-snowflake-1780", "spring-totem-1779"],
    2018: ["lunar-eon-1788", "worthy-plant-1787", "astral-oath-1786", "comfy-wind-1785", "peach-pine-1784"],
    2019: ["gentle-elevator-1793", "swept-armadillo-1792", "vague-capybara-1791", "glad-sponge-1790", "ethereal-sun-1789"],
    2020: ["peach-aardvark-1798", "hearty-moon-1797", "autumn-leaf-1796", "pleasant-donkey-1795", "vibrant-snowflake-1794"],
    2021: ["soft-gorge-1803", "silvery-flower-1802", "floral-flower-1801", "icy-rain-1800", "fresh-snow-1799"],
    2022: ["fanciful-frog-1808", "light-jazz-1807", "deep-energy-1806", "cosmic-galaxy-1805", "dutiful-music-1804"],
    2023: ["dark-durian-1813", "prime-oath-1812", "devout-surf-1811", "fanciful-salad-1810", "atomic-flower-1809"],
    2024: ["lemon-sunset-1818", "quiet-meadow-1817", "radiant-aardvark-1816", "wild-pine-1815", "wise-surf-1814"],
}

# --------------------------------------------------------------------------- #
# Plot styling (matches the varying-forecast figure for a consistent paper)
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


def run_csv(run):
    return f"{ROOT_DIR}/{run}_ForecastModel_UKMO_EnsembleNum0/precision_recall.csv"


def pr_avg(run_list):
    """Average several runs' PR curves element-wise by threshold rank.

    The exports are ordered by decision threshold (ascending) but differ in
    length, so we truncate to the shortest run and average the real recall and
    precision values at each rank. No interpolation onto a synthetic grid is
    performed: every returned point is the mean of five real measured points.
    """
    dfs = [pd.read_csv(run_csv(r)) for r in run_list]
    n = min(len(d) for d in dfs)
    recall = np.mean([d["recall"].values[:n] for d in dfs], axis=0)
    precision = np.mean([d["precision"].values[:n] for d in dfs], axis=0)
    return pd.DataFrame({"recall": recall, "precision": precision})


# --------------------------------------------------------------------------- #
# Compute per-year mean curves
# --------------------------------------------------------------------------- #
years = sorted(RF_RUNS)
year_curves = {}
for y in years:
    avg = pr_avg(RF_RUNS[y])
    year_curves[y] = avg
    if SAVE_YEARLY_CSV:
        avg.to_csv(f"{SAVE_DIR}/rf_{y}.csv", index=False)

# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
# Refined qualitative 9-colour palette (seaborn "deep"); distinguishable but
# softer / more designed than the default matplotlib cycle.
colors = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974",
]

fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)

# A thin white stroke around each line keeps the upper curve readable where the
# nine curves overlap.
outline = [pe.Stroke(linewidth=3.4, foreground="white"), pe.Normal()]

for y, color in zip(years, colors):
    # Sort each curve's real points by recall so the noisy high-threshold tail
    # reads as a clean line rather than a zig-zag (reorders real points only;
    # no interpolation / fabricated data).
    curve = year_curves[y].sort_values("recall")
    ax.plot(
        curve["recall"], curve["precision"],
        color=color, linewidth=2.0, label=str(y),
        solid_capstyle="round", path_effects=outline,
    )

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Random Forest Temporal Cross-validation")
ax.grid(True, linestyle="-", linewidth=0.4, alpha=0.2)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

# Legend outside the axes (right) so it never overlaps the curves.
ax.legend(
    title="Test year",
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
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
