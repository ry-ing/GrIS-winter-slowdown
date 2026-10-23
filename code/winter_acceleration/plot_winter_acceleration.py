# Plots for analyze_winter_acceleration.py

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import config

DATA_DIR = os.path.join(config.DATA_DIR, 'winter_acceleration')
OUTPUT_DIR = os.path.join(config.FIG_DIR, 'winter_acceleration')

BASELINE_YEARS = [2016, 2017, 2018, 2019, 2020]
POST_EVENT_YEARS = [2021, 2022, 2023, 2024]

EXAMPLE_ROIS = [1, 4, 7, 10]   # spread across the ROIs for the diagnostic figure

FS_LABEL = 11
FS_TICK = 9

os.makedirs(OUTPUT_DIR, exist_ok=True)

ds = xr.open_dataset(os.path.join(DATA_DIR, 'winter_timing_and_acceleration.nc'))
roi_summary_df = pd.read_csv(os.path.join(DATA_DIR, 'roi_winter_acceleration_summary.csv'),
                             parse_dates=['start_date', 'end_date'])
curves_df = pd.read_csv(os.path.join(DATA_DIR, 'roi_smoothed_curves.csv'), parse_dates=['time'])


# maps
baseline = ds['pct_acceleration'].sel(year=BASELINE_YEARS).mean(dim='year', skipna=True)
post = ds['pct_acceleration'].sel(year=POST_EVENT_YEARS).mean(dim='year', skipna=True)
change = post - baseline

vmax = float(np.nanpercentile(np.abs(baseline.values), 95))
fig, ax = plt.subplots(figsize=(6, 5.5))
im = ax.pcolormesh(baseline.x, baseline.y, baseline, cmap='YlGnBu', vmin=0, vmax=vmax)
ax.set_title('Mean winter acceleration (%)\nbaseline 2016-2020', fontsize=FS_LABEL)
ax.set_aspect('equal')
ax.set_xlabel('Easting (m, EPSG:3413)', fontsize=FS_TICK)
ax.set_ylabel('Northing (m, EPSG:3413)', fontsize=FS_TICK)
ax.tick_params(labelsize=FS_TICK)
fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, label='% acceleration (autumn min -> spring transition)')
fig.tight_layout()
out = os.path.join(OUTPUT_DIR, 'spatial_map_pct_acceleration_baseline.png')
fig.savefig(out, dpi=250, bbox_inches='tight')
plt.close(fig)

vmax_c = float(np.nanpercentile(np.abs(change.values), 95))
fig, ax = plt.subplots(figsize=(6, 5.5))
im = ax.pcolormesh(change.x, change.y, change, cmap='RdBu_r', vmin=-vmax_c, vmax=vmax_c)
ax.set_title('Change in winter acceleration (%)\npost-event (2021-2024) minus baseline', fontsize=FS_LABEL)
ax.set_aspect('equal')
ax.set_xlabel('Easting (m, EPSG:3413)', fontsize=FS_TICK)
ax.set_ylabel('Northing (m, EPSG:3413)', fontsize=FS_TICK)
ax.tick_params(labelsize=FS_TICK)
fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, label='Δ % acceleration')
fig.tight_layout()
out = os.path.join(OUTPUT_DIR, 'spatial_map_pct_acceleration_change_post_vs_baseline.png')
fig.savefig(out, dpi=250, bbox_inches='tight')
plt.close(fig)


# check the detection on a few ROIs
n = len(EXAMPLE_ROIS)
fig, axes = plt.subplots(n, 1, figsize=(9, 2.3 * n), sharex=True)

for i in range(n):
    lbl = EXAMPLE_ROIS[i]
    ax = axes[i]
    sub = curves_df[curves_df['roi_label'] == lbl].sort_values('time')
    ax.plot(pd.to_datetime(sub['time']), sub['velocity_smooth'], color='#1a6fbf', lw=1.1,
            label='Smoothed velocity', zorder=3)

    roi_sub = roi_summary_df[roi_summary_df['roi_label'] == lbl]
    for idx, row in roi_sub.iterrows():
        if pd.isna(row['start_date']) or pd.isna(row['end_date']):
            continue
        t0 = pd.Timestamp(row['start_date'])
        t1 = pd.Timestamp(row['end_date'])
        # only label the first one in the legend
        if idx == roi_sub.index[0]:
            start_label = 'Winter start (autumn min)'
            end_label = 'Winter end (spring transition)'
        else:
            start_label = None
            end_label = None
        ax.scatter([t0], [row['v_start']], color='#c0392b', s=35, zorder=5, marker='v', label=start_label)
        ax.scatter([t1], [row['v_end']], color='#27ae60', s=35, zorder=5, marker='^', label=end_label)
        if np.isfinite(row['linear_slope_m_yr_per_day']):
            n_days = float((t1 - t0).days)
            ax.plot([t0, t1], [row['v_start'], row['v_start'] + row['linear_slope_m_yr_per_day'] * n_days],
                    color='black', lw=1.3, ls='--', zorder=4)

    ax.text(0.01, 0.92, f'L{lbl}', transform=ax.transAxes, fontsize=FS_LABEL, fontweight='bold', va='top')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linewidth=0.6, color='#888888', linestyle=':')
    ax.set_ylabel('Velocity (m/yr)', fontsize=FS_TICK)
    if i == 0:
        ax.legend(fontsize=7, loc='upper right', ncol=3, framealpha=0.9)

axes[-1].set_xlabel('Date', fontsize=FS_LABEL)
fig.tight_layout()
out = os.path.join(OUTPUT_DIR, 'roi_diagnostic_examples.png')
fig.savefig(out, dpi=250, bbox_inches='tight')
plt.close(fig)


# % acceleration for each ROI
roi_labels = sorted(roi_summary_df['roi_label'].unique())
years = sorted(roi_summary_df['winter_start_year'].unique())
colors = []
for y in years:
    if y in BASELINE_YEARS:
        colors.append('#AAAAAA')
    else:
        colors.append('#1a6fbf')

nrow = int(np.ceil(len(roi_labels) / 2))
fig, axes = plt.subplots(nrow, 2, figsize=(9, 1.9 * nrow), sharex=True)
axes = axes.ravel()

for i in range(len(roi_labels)):
    lbl = roi_labels[i]
    ax = axes[i]
    sub = roi_summary_df[roi_summary_df['roi_label'] == lbl].set_index('winter_start_year')
    vals = [sub['pct_acceleration'].get(y, np.nan) for y in years]
    ax.bar(range(len(years)), vals, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', lw=0.8)
    ax.text(0.03, 0.9, f'L{lbl}', transform=ax.transAxes, fontsize=FS_LABEL, fontweight='bold', va='top')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linewidth=0.6, color='#888888', linestyle=':')
    # year labels on the bottom row only
    if i >= len(roi_labels) - 2:
        ax.set_xticks(range(len(years)))
        ax.set_xticklabels([f'{y}/{str(y+1)[2:]}' for y in years], rotation=45, ha='right', fontsize=7)

for j in range(len(roi_labels), len(axes)):
    axes[j].set_visible(False)

fig.text(0.02, 0.5, 'Winter acceleration (%): autumn min -> spring transition',
         rotation='vertical', va='center', fontsize=FS_LABEL)
fig.tight_layout(rect=[0.04, 0.0, 1.0, 1.0])
out = os.path.join(OUTPUT_DIR, 'roi_pct_acceleration_by_year.png')
fig.savefig(out, dpi=250, bbox_inches='tight')
plt.close(fig)


# timing histograms
start_vals = pd.to_datetime(ds['start_date'].values.ravel())
end_vals = pd.to_datetime(ds['end_date'].values.ravel())
start_doy = start_vals.dayofyear[~start_vals.isna()]
end_doy = end_vals.dayofyear[~end_vals.isna()]

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
axes[0].hist(start_doy, bins=30, color='#c0392b', alpha=0.8)
axes[0].set_title('Start of winter (autumn minimum)', fontsize=FS_LABEL)
axes[1].hist(end_doy, bins=30, color='#27ae60', alpha=0.8)
axes[1].set_title('End of winter (spring transition)', fontsize=FS_LABEL)
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linewidth=0.6, color='#888888', linestyle=':')
    ax.set_xlabel('Day of year', fontsize=FS_TICK)
    ax.set_ylabel('Pixel-winters', fontsize=FS_TICK)

fig.tight_layout()
out = os.path.join(OUTPUT_DIR, 'histogram_winter_timing.png')
fig.savefig(out, dpi=250, bbox_inches='tight')
plt.close(fig)

