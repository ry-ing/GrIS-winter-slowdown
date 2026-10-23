# Flowline anomaly vs elevation and thickness, run prep_flowline_data.py first

import os
import sys
import numpy as np
import pandas as pd
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import config

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = os.path.join(config.DATA_DIR, 'scatter_plots')
OUTPUT_DIR = config.FIG_DIR
SELECTED_ROI_LABELS = config.ROI_LABELS

FS_LABEL = 12
FS_TICK = 10
FS_LEGEND = 10
FS_LETTER = 14

# rolling window (points)
ROLL_WINDOW = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

# read the flowlines
flowlines = []
for lbl in SELECTED_ROI_LABELS:
    df = pd.read_csv(os.path.join(DATA_DIR, f'flowline_ROI_{lbl}.csv'))
    df['roi'] = lbl
    flowlines.append(df)
all_flow = pd.concat(flowlines, ignore_index=True)

fig, axes = plt.subplots(2, 1, figsize=(7.5, 6), gridspec_kw={'hspace': 0.35})


# (a) W1 anomaly vs surface elevation
ax = axes[0]
bins_a = np.arange(0, 1600, 50)
bin_centers_a = bins_a[:-1] + 25

# each ROI, sorted by elevation and smoothed
for lbl in SELECTED_ROI_LABELS:
    df_roi = all_flow[all_flow['roi'] == lbl].dropna(subset=['surface_elev', 'W1_anom'])
    df_roi = df_roi.sort_values('surface_elev')
    smooth_y = df_roi['W1_anom'].rolling(window=ROLL_WINDOW, center=True, min_periods=1).mean()
    ax.plot(df_roi['surface_elev'], smooth_y, color='dodgerblue', alpha=0.7, linewidth=1.5, zorder=2)

# mean of all ROIs in 50 m bins, lightly smoothed
valid_a = all_flow.dropna(subset=['surface_elev', 'W1_anom']).copy()
valid_a['elev_bin'] = pd.cut(valid_a['surface_elev'], bins=bins_a)
mean_a = valid_a.groupby('elev_bin', observed=False)['W1_anom'].mean()
count_a = valid_a.groupby('elev_bin', observed=False)['W1_anom'].count()
smooth_mean_a = mean_a.rolling(window=2, center=True, min_periods=1).mean()
smooth_mean_a = smooth_mean_a.where(count_a > 4)
smooth_mean_a[bin_centers_a < 300] = np.nan

ax.plot(bin_centers_a, smooth_mean_a.values, color='#08306b', linewidth=3.0,
        label='Multi-ROI Mean', zorder=4,
        path_effects=[pe.withStroke(linewidth=4, foreground='white')])
ax.legend(loc='lower right', fontsize=FS_LEGEND)

ax.set_xlim(0, 1400)
ymin = np.nanpercentile(all_flow['W1_anom'], 1) - 5
ax.set_ylim(ymin, 10)

ax.set_xlabel('Surface elevation (m a.s.l.)', fontsize=FS_LABEL)
ax.set_ylabel('Winter 2021/22 Velocity anomaly (%)', fontsize=FS_LABEL)
ax.tick_params(labelsize=FS_TICK)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.35, linewidth=0.6, color='#888888', linestyle=':')
ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.5, zorder=1)
# move the y label down so it covers both panels
ax.yaxis.set_label_coords(-0.11, -0.175)
ax.text(0.02, 0.98, '(a)', transform=ax.transAxes, fontsize=FS_LETTER, fontweight='bold', va='top', ha='left',
        path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])


# (b) W1 anomaly vs ice thickness
ax = axes[1]
max_thick = all_flow['ice_thickness'].max()
bins_b = np.arange(0, max_thick + 50, 50)
bin_centers_b = bins_b[:-1] + 25

for lbl in SELECTED_ROI_LABELS:
    df_roi = all_flow[all_flow['roi'] == lbl].dropna(subset=['ice_thickness', 'W1_anom'])
    df_roi = df_roi.sort_values('ice_thickness')
    smooth_y = df_roi['W1_anom'].rolling(window=ROLL_WINDOW, center=True, min_periods=1).mean()
    ax.plot(df_roi['ice_thickness'], smooth_y, color='mediumseagreen', alpha=0.7, linewidth=1.5, zorder=2)

valid_b = all_flow.dropna(subset=['ice_thickness', 'W1_anom']).copy()
valid_b['thick_bin'] = pd.cut(valid_b['ice_thickness'], bins=bins_b)
mean_b = valid_b.groupby('thick_bin', observed=False)['W1_anom'].mean()
count_b = valid_b.groupby('thick_bin', observed=False)['W1_anom'].count()
smooth_mean_b = mean_b.rolling(window=2, center=True, min_periods=1).mean()
# no smoothing for bins with 3 or fewer points
smooth_mean_b[count_b <= 3] = mean_b[count_b <= 3]
smooth_mean_b[bin_centers_b < 200] = np.nan

ax.plot(bin_centers_b, smooth_mean_b.values, color='#00441b', linewidth=3.0,
        label='Multi-ROI Mean', zorder=4,
        path_effects=[pe.withStroke(linewidth=4, foreground='white')])
ax.legend(loc='lower right', fontsize=FS_LEGEND)

ax.set_xlim(0, 1600)
ax.set_ylim(ymin, 10)
ax.set_xlabel('Ice thickness (m)', fontsize=FS_LABEL)
ax.tick_params(labelsize=FS_TICK)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.35, linewidth=0.6, color='#888888', linestyle=':')
ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.5, zorder=1)
ax.text(0.02, 0.98, '(b)', transform=ax.transAxes, fontsize=FS_LETTER, fontweight='bold', va='top', ha='left',
        path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

out = os.path.join(OUTPUT_DIR, 'Fig05_elevation_thickness.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close(fig)
