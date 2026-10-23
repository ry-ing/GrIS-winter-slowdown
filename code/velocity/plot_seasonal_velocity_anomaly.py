# Summer, winter and annual velocity anomaly at the land-terminating ROIs
# from the ROI timeseries (all the points, not the rolling mean)

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import config

warnings.filterwarnings('ignore')

PREP_DIR = os.path.join(config.DATA_DIR, 'velocity')
OUTPUT_DIR = config.FIG_DIR

YEARS = list(range(2016, 2025))
BASELINE_YEARS = [2016, 2017, 2018, 2019, 2020]
MIN_POINTS = 3   # velocity points needed in a season

POST_COLORS = {
    2021: '#0C2C84',
    2022: '#1D91C0',
    2023: '#41B6C4',
    2024: '#C7E9B4',
}
PRE_COLOR = '#AAAAAA'
FONT = 28
FONT_SM = 24
FONT_PL = 34


# windows for each year (start, end)
windows = {}
for yr in YEARS:
    windows[('summer', yr)] = (pd.Timestamp(yr, 5, 1), pd.Timestamp(yr, 8, 31))
    windows[('winter', yr)] = (pd.Timestamp(yr, 11, 1), pd.Timestamp(yr + 1, 4, 30))
    windows[('annual', yr)] = (pd.Timestamp(yr, 5, 1), pd.Timestamp(yr + 1, 4, 30))   # summer + the following winter

# velocity anomaly (%) for each ROI, season and year
anom = {}
for season in ['summer', 'winter', 'annual']:
    anom[season] = np.full((len(config.ROI_LABELS), len(YEARS)), np.nan)

for i in range(len(config.ROI_LABELS)):
    lbl = config.ROI_LABELS[i]
    ts = pd.read_csv(os.path.join(PREP_DIR, f'ROI_{lbl}_timeseries.csv'), parse_dates=['date'])
    ts = ts.dropna(subset=['velocity'])
    for season in ['summer', 'winter', 'annual']:
        # mean of the velocity points (ROI spatial medians) in each season
        mean_v = {}
        for yr in YEARS:
            w0, w1 = windows[(season, yr)]
            v = ts.loc[ts['date'].between(w0, w1), 'velocity']
            if len(v) >= MIN_POINTS:
                mean_v[yr] = v.mean()
            else:
                mean_v[yr] = np.nan

        # 2016-2020 mean
        base = np.nanmean([mean_v[yr] for yr in BASELINE_YEARS])
        for j in range(len(YEARS)):
            anom[season][i, j] = (mean_v[YEARS[j]] / base - 1) * 100

# bars = mean of the 10 ROIs, error bars = standard error
mean = {}
se = {}
for season in anom:
    mean[season] = np.nanmean(anom[season], axis=0)
    n = np.isfinite(anom[season]).sum(axis=0)
    se[season] = np.nanstd(anom[season], axis=0, ddof=1) / np.sqrt(n)

print('mean anomaly across ROIs (%), summer / winter / annual')
for j in range(len(YEARS)):
    yr = YEARS[j]
    print(yr, round(mean['summer'][j], 1), round(mean['winter'][j], 1), round(mean['annual'][j], 1))


# figure
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': FONT,
    'axes.labelsize': FONT,
    'xtick.labelsize': FONT,
    'ytick.labelsize': FONT,
    'legend.fontsize': FONT_SM,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
})

colors = []
for yr in YEARS:
    if yr in POST_COLORS:
        colors.append(POST_COLORS[yr])
    else:
        colors.append(PRE_COLOR)

fig, axes = plt.subplots(3, 1, figsize=(18, 15), facecolor='white')
fig.subplots_adjust(left=0.16, right=0.97, top=0.97, bottom=0.09, hspace=0.75)

panels = [('summer', 'Summer velocity\nanomaly (%)', [str(yr) for yr in YEARS], 'Summer season (1 May – 31 Aug)'),
          ('winter', 'Winter velocity\nanomaly (%)', [f'{yr}/{str(yr + 1)[2:]}' for yr in YEARS],
           'Winter season (1 Nov – 30 Apr)'),
          ('annual', 'Annual velocity\nanomaly (%)', [f'{yr}/{str(yr + 1)[2:]}' for yr in YEARS],
           'Year (1 May – 30 Apr)')]
letters = ['(a)', '(b)', '(c)']

x = np.arange(len(YEARS))
for k in range(3):
    season, ylabel, xlabels, xlabel = panels[k]
    ax = axes[k]
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, axis='y', alpha=0.20, linewidth=0.5, color='#999999')

    # range of the 2016-2020 years
    base_vals = mean[season][:len(BASELINE_YEARS)]
    ax.axhspan(base_vals.min(), base_vals.max(), color='#DDDDDD', alpha=0.6, zorder=0)

    ax.bar(x, mean[season], width=0.7, color=colors, edgecolor='black', linewidth=0.8, zorder=2)
    ax.errorbar(x, mean[season], yerr=se[season], fmt='none', ecolor='black', capsize=6, linewidth=1.2, zorder=3)
    ax.axhline(0, color='black', linewidth=1.0, zorder=1)

    ax.axvline(4.5, color='#D62728', linestyle='--', linewidth=1.2, alpha=0.8, zorder=1)
    ax.text(4.6, 0.97, 'August 2021', transform=ax.get_xaxis_transform(), color='#D62728', fontsize=FONT_SM,
            style='italic', va='top')

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=40, ha='right')
    ax.set_xlabel(xlabel, fontsize=FONT)
    ax.set_ylabel(ylabel, fontsize=FONT)
    ax.text(0.02, 0.97, letters[k], transform=ax.transAxes, fontsize=FONT_PL, fontweight='bold', va='top')

# same y range in all panels
y_lo = min(ax.get_ylim()[0] for ax in axes)
y_hi = max(ax.get_ylim()[1] for ax in axes)
# room above the bars for the August 2021 label
y_hi = y_hi + 0.25 * (y_hi - y_lo)
for ax in axes:
    ax.set_ylim(y_lo, y_hi)

p = os.path.join(OUTPUT_DIR, 'FigS4_seasonal_velocity_anomaly.png')
fig.savefig(p, dpi=300, bbox_inches='tight')
plt.close(fig)
