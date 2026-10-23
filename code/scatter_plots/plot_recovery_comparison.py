# Recovery vs slowdown at each ROI, run prepare_vel_scatter_data.py first

import os
import sys
import numpy as np
import pandas as pd
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
import xarray as xr
from scipy import stats

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import config

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = os.path.join(config.DATA_DIR, 'scatter_plots')
OUTPUT_DIR = config.FIG_DIR

MIN_W1_SLOWDOWN_PCT = 1.0   # %

PIX_W1_MIN = 7.5       # % - pixels need W1 below -7.5 %
BIN_WIDTH = 2.5        # % - width of the W1 bins in panel (c)
BIN_MIN = 7.5
BIN_MAX = 40.0
MIN_PIX_PER_BIN = 200
RECOVERY_THRESHOLD_PCT = -1.0   # recovered when anomaly >= -1 %, i.e. >= 99 % of baseline

FS_LABEL = 11
FS_TICK = 9
FS_ANNOT = 9
FS_LEGEND = 9
FS_LETTER = 13

# colours
cmap = plt.cm.YlGnBu
BAR_COLORS = {1: cmap(0.15), 2: cmap(0.38), 3: cmap(0.62), 5: cmap(0.87)}
BAR_LABELS = {1: '1', 2: '2', 3: '3', 5: '>4'}

# colours for panel (b)
COLOR_NORMAL = '#31C950'
COLOR_CENSORED = cmap(0.87)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# winters to recover for each ROI
roi_df = pd.read_csv(os.path.join(DATA_DIR, 'roi_recovery.csv'))
years = []
for i, row in roi_df.iterrows():
    n = 0
    for w in ['W1', 'W2', 'W3', 'W4']:
        if row[f'{w}_anom_pct'] < RECOVERY_THRESHOLD_PCT:
            n += 1
        else:
            break
    if n == 4:
        n = 5
    years.append(n)
roi_df['years_to_recover'] = years

# pixel anomaly grids
grids = {}
for w in [1, 2, 3, 4]:
    ds = xr.open_dataset(os.path.join(DATA_DIR, f'anomaly_W{w}.nc'))
    grids[w] = ds[f'anomaly_W{w}'].values.copy()
    ds.close()
valid = np.isfinite(grids[1]) & np.isfinite(grids[2]) & np.isfinite(grids[3]) & np.isfinite(grids[4])
mask = (grids[1] < -PIX_W1_MIN) & valid
w1_pix = np.abs(grids[1][mask])
w2 = grids[2][mask]
w3 = grids[3][mask]
w4 = grids[4][mask]

# recovery for each pixel
dur_pix = np.full(len(w1_pix), 5.0)
dur_pix[w4 >= RECOVERY_THRESHOLD_PCT] = 3
dur_pix[w3 >= RECOVERY_THRESHOLD_PCT] = 2
dur_pix[w2 >= RECOVERY_THRESHOLD_PCT] = 1
n_total = len(w1_pix)
print(n_total, 'pixels')

# bin the pixels by W1 for panel (c)
bins = np.arange(BIN_MIN, BIN_MAX + BIN_WIDTH, BIN_WIDTH)
bin_centers = (bins[:-1] + bins[1:]) / 2
n_bins = len(bin_centers)
props = {}
for d in [1, 2, 3, 5]:
    props[d] = np.full(n_bins, np.nan)
n_count = np.zeros(n_bins, dtype=int)
for i in range(n_bins):
    sel = (w1_pix >= bins[i]) & (w1_pix < bins[i + 1])
    n = int(sel.sum())
    n_count[i] = n
    if n >= MIN_PIX_PER_BIN:
        for d in [1, 2, 3, 5]:
            props[d][i] = np.sum(dur_pix[sel] == d) / n
valid_bins = n_count >= MIN_PIX_PER_BIN

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={'wspace': 0.2})


# (a) baseline velocity vs recovery slope
ax = axes[0]
plot_df = roi_df[(roi_df['W1_anom_pct'] < -MIN_W1_SLOWDOWN_PCT) & roi_df['recovery_slope'].notna()].copy()

for i, row in plot_df.iterrows():
    lbl = int(row['roi_label'])
    x = row['baseline_vel']
    y = row['recovery_slope']
    se = row['recovery_slope_se']
    if not np.isfinite(se):
        se = 0
    ax.errorbar(x, y, yerr=se, fmt='o', color='dodgerblue', markersize=7,
                capsize=3, elinewidth=1.2, markeredgecolor='k',
                markeredgewidth=0.7, zorder=4)
    ax.annotate(f'L{lbl}', xy=(x, y), xytext=(6, 1),
                textcoords='offset points', fontsize=FS_ANNOT,
                fontweight='bold', color='#1a4f8a', va='center')

xf = plot_df['baseline_vel'].values
yf = plot_df['recovery_slope'].values
res = stats.linregress(xf, yf)
xl = np.linspace(xf.min(), xf.max(), 200)
ax.plot(xl, res.slope * xl + res.intercept, color='black', linewidth=1.3, linestyle='--', alpha=0.8, zorder=3)
ax.text(0.97, 0.05, f'R²={res.rvalue**2:.2f},  p={res.pvalue:.2f}',
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=FS_ANNOT, color='#333333',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='k', alpha=0.75), zorder=5)

ax.set_xlabel('Pre-event baseline winter velocity (m yr$^{-1}$)', fontsize=FS_LABEL)
ax.set_ylabel('Recovery slope (m yr$^{-1}$ per winter)', fontsize=FS_LABEL)
ax.tick_params(labelsize=FS_TICK)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.3, linewidth=0.6, color='#888888', linestyle=':')
ax.text(0.02, 0.98, '(a)', transform=ax.transAxes, fontsize=FS_LETTER, fontweight='bold', va='top', ha='left',
        path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])


# (b) recovery duration vs W1 anomaly
ax = axes[1]

plot_df_b = roi_df[roi_df['W1_anom_pct'] < -MIN_W1_SLOWDOWN_PCT].copy()
# not recovered (5) is plotted at 4.5
plot_df_b['yv_plot'] = plot_df_b['years_to_recover'].astype(float)
plot_df_b.loc[plot_df_b['years_to_recover'] == 5, 'yv_plot'] = 4.5
plot_df_b['xv'] = plot_df_b['W1_anom_pct'].abs()
xv = plot_df_b['xv'].values
yv = plot_df_b['yv_plot'].values

# x range first
x_pad = (xv.max() - xv.min()) * 0.12
ax.set_xlim(max(0, xv.min() - x_pad), xv.max() + x_pad)

for i, row in plot_df_b.iterrows():
    if row['years_to_recover'] == 5:
        ax.scatter(row['xv'], row['yv_plot'], s=90, color=COLOR_CENSORED, marker='^',
                   edgecolor='black', linewidth=0.7, zorder=4, alpha=0.92)
    else:
        ax.scatter(row['xv'], row['yv_plot'], s=90, color=COLOR_NORMAL, marker='o',
                   edgecolor='black', linewidth=0.7, zorder=4, alpha=0.92)

# labels, alternate above/below
for yrow, group in plot_df_b.groupby('yv_plot'):
    group = group.sort_values('xv')
    k = 0
    for i, row in group.iterrows():
        if k % 2 == 0:
            dy = 0.30
        else:
            dy = -0.30
        dx = 0.5
        ha = 'left'
        if row['xv'] + dx + 2.0 > ax.get_xlim()[1]:
            dx = -0.5
            ha = 'right'
        ax.text(row['xv'] + dx, row['yv_plot'] + dy, f"L{int(row['roi_label'])}",
                fontsize=FS_ANNOT, fontweight='bold', color='#1a1a1a', va='center', ha=ha)
        k += 1

# regression
res = stats.linregress(xv, yv)   
xl = np.linspace(xv.min(), xv.max(), 200)
ax.plot(xl, res.slope * xl + res.intercept, color='black', linewidth=1.3, linestyle='--', alpha=0.8, zorder=3)

# spearman rank
rho, p_rho = stats.spearmanr(xv, plot_df_b['years_to_recover'].values)
ax.text(0.97, 0.05, f'ρ={rho:.2f},  p={p_rho:.2f}',
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=FS_ANNOT, color='#333333',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='k', alpha=0.75), zorder=5)

ax.set_ylim(0.5, 5.2)
ax.set_yticks([1, 2, 3, 4.5])
ax.set_yticklabels(['1', '2', '3', '>4'])
ax.set_xlabel('Winter 2021/22 velocity anomaly magnitude (%)', fontsize=FS_LABEL)
ax.set_ylabel('Recovery duration (winters)', fontsize=FS_LABEL)
ax.tick_params(labelsize=FS_TICK)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.3, linewidth=0.6, color='#888888', linestyle=':')
ax.text(0.02, 0.98, '(b)', transform=ax.transAxes, fontsize=FS_LETTER, fontweight='bold', va='top', ha='left',
        path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])


# pixel proportions
ax = axes[2]

bottoms = np.zeros(n_bins)
for d in [1, 2, 3, 5]:
    p = np.where(valid_bins, props[d], 0.0)
    ax.bar(bin_centers, p, width=BIN_WIDTH, bottom=bottoms, color=BAR_COLORS[d], label=BAR_LABELS[d],
           edgecolor='black', linewidth=0.4)
    bottoms = bottoms + p

# n pixels
for i in range(n_bins):
    if valid_bins[i]:
        n = n_count[i]
        if n >= 1000:
            lbl = f'{n // 1000}k'
        else:
            lbl = str(n)
        ax.text(bin_centers[i], 0.045, lbl, ha='center', va='top', fontsize=6.5, color='black', clip_on=False)

patches = []
for d in [1, 2, 3, 5]:
    patches.append(Patch(facecolor=BAR_COLORS[d], edgecolor='black', linewidth=0.4, label=BAR_LABELS[d]))
leg = ax.legend(handles=patches, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.0),
                frameon=False, fontsize=FS_LEGEND, columnspacing=0.8, handlelength=1.2,
                title='Recovery duration (winters)', title_fontsize=FS_LEGEND)
leg.get_title().set_fontweight('normal')

# x limit
last_bin = np.flatnonzero(valid_bins).max()
ax.set_xlim(BIN_MIN, bins[last_bin + 1])
ax.set_ylim(0, 1.0)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['0', '25', '50', '75', '100'])
ax.set_xlabel('Winter 2021/22 velocity anomaly magnitude (%)', fontsize=FS_LABEL)
ax.set_ylabel('Proportion of pixels (%)', fontsize=FS_LABEL)
ax.tick_params(labelsize=FS_TICK)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.3, linewidth=0.6, color='#888888', linestyle=':')
ax.text(0.02, 0.98, '(c)', transform=ax.transAxes, fontsize=FS_LETTER, fontweight='bold', va='top', ha='left',
        path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

plt.subplots_adjust(top=0.83)
out = os.path.join(OUTPUT_DIR, 'Fig08_scatter_plots.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close(fig)
