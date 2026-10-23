# Map and velocity timeseries for the land-terminating ROIs L1-L10

import os
import sys
import json
import numpy as np
import xarray as xr
import geopandas as gpd
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
from matplotlib.patches import Circle, Patch
from matplotlib.colors import LightSource
import cartopy.crs as ccrs

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import config

PREP_DIR = os.path.join(config.DATA_DIR, 'velocity')
OUTPUT_DIR = config.FIG_DIR
MANUAL_WINDOWS_PATH = os.path.join(PREP_DIR, 'winter_windows.json')
WINTER_MEDIAN_CSV_DIR = os.path.join(PREP_DIR, 'winter_median_csvs')

SELECTED_ROI_LABELS = config.ROI_LABELS

# label offsets
ROI_LABEL_FRAC = 0.14
ROI_LABEL_DY_FRAC = -0.025
ROI_LABEL_RADIUS_FRAC = 0.030

CMAP_ANOM = 'YlGnBu'
NORM_ANOM = mcolors.Normalize(vmin=-20, vmax=0)
OCEAN_COLOR = '#AFCED9'

EVENT_DATE = pd.to_datetime('2021-08-14')

PROJ_3413 = ccrs.Stereographic(central_latitude=90, central_longitude=-45, true_scale_latitude=70)

# figure layout, in figure fractions
MAPS_X0 = 0.02     # map column
MAPS_X1 = 0.53
TS_X0 = 0.52       # timeseries column
TS_X1 = 0.97
FIG_Y0 = 0.10      # extra space at the bottom for the colourbar
FIG_Y1 = 0.97

# font sizes
FS_LABEL = 28
FS_TICK = 26
FS_ANNOT = 24
FS_CBAR = 28
FS_LETTER = 30
FS_ELEV = 20
FS_MAP_ROI_LABEL = FS_ANNOT - 2

# y padding
TS_YLIM_LOWER_PAD_FRAC = 0.08
TS_YLIM_UPPER_PAD_FRAC = 0.12

# label shift (days)
PCT_LABEL_DAYS_OFFSET = -7


# slice y
def yslice(da, ylo, yhi):
    if float(da.y[0]) > float(da.y[-1]):
        return da.sel(y=slice(yhi, ylo))
    return da.sel(y=slice(ylo, yhi))


# % change label next to a winter median line
def annotate_pct(ax, pct, t_end, lmed):
    if pct < 0:
        txt_color = 'darkred'
        edge_color = 'crimson'
    else:
        txt_color = 'darkgreen'
        edge_color = 'darkgreen'
    ann = ax.annotate(f'{pct:+.1f}%',
                      xy=(t_end + pd.Timedelta(days=PCT_LABEL_DAYS_OFFSET), lmed),
                      xytext=(0, -10), textcoords='offset points',
                      color=txt_color, fontsize=FS_ANNOT - 2,
                      va='top', ha='left', zorder=10,
                      annotation_clip=False, clip_on=False,
                      bbox=dict(facecolor='white', alpha=0.45, edgecolor=edge_color,
                                boxstyle='round,pad=0.15', linewidth=0.5))
    ann.get_bbox_patch().set_clip_on(False)


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WINTER_MEDIAN_CSV_DIR, exist_ok=True)

# winter windows
with open(MANUAL_WINDOWS_PATH) as f:
    windows_json = json.load(f)
manual_windows = {}
for k in windows_json:
    manual_windows[int(k)] = windows_json[k]

anom_da = xr.open_dataarray(os.path.join(PREP_DIR, 'spatial_anomaly_w1.nc'))
recovery_da = xr.open_dataarray(os.path.join(PREP_DIR, 'recovery_time.nc'))
mask_da = xr.open_dataarray(os.path.join(config.CACHE_DIR, 'ice_mask.nc'))
surf_da = xr.open_dataarray(os.path.join(config.CACHE_DIR, 'bed_surface.nc'))
valid_mask_da = xr.open_dataarray(os.path.join(PREP_DIR, 'valid_mask.nc'))

rois = gpd.read_file(config.ROI_SHAPEFILE_PATH).to_crs('EPSG:3413')
rois['label'] = rois['id'].astype(int)
rois = rois[rois['label'].isin(SELECTED_ROI_LABELS)].sort_values('label').reset_index(drop=True)

n_ts = len(SELECTED_ROI_LABELS)
fig_w = 32.0
fig_h = max(24.0, n_ts * 2.8)
fig = plt.figure(figsize=(fig_w, fig_h))

maps_w_in = (MAPS_X1 - MAPS_X0) * fig_w
maps_h_in = (FIG_Y1 - FIG_Y0) * fig_h


# (a) map, sized to the study extent
xmin, xmax, ymin, ymax = config.STUDY_EXTENT
W_m = xmax - xmin
H_m = ymax - ymin
aspect = H_m / W_m

# map size
map_w_in = maps_w_in
map_h_in = map_w_in * aspect
if map_h_in > maps_h_in:
    map_h_in = maps_h_in
    map_w_in = map_h_in / aspect

# centre it in the map column
map_x0_fig = MAPS_X0 + (maps_w_in - map_w_in) / 2 / fig_w
map_y0_fig = FIG_Y0 + (maps_h_in - map_h_in) / 2 / fig_h

ax = fig.add_axes([map_x0_fig, map_y0_fig, map_w_in / fig_w, map_h_in / fig_h], projection=PROJ_3413)
ax.set_extent([xmin, xmax, ymin, ymax], crs=PROJ_3413)
ax.set_facecolor(OCEAN_COLOR)

# hillshade
surf_hs = yslice(surf_da.sel(x=slice(xmin - 5000, xmax + 5000)), ymin - 5000, ymax + 5000)
surf_vals = surf_hs.values
land = np.where(np.isfinite(surf_vals), surf_vals > 1.0, False)
shade = LightSource(azdeg=315, altdeg=45).hillshade(np.where(land, surf_vals, 0.0), vert_exag=10, dx=150, dy=150)
sh_ext = [float(surf_hs.x.min()), float(surf_hs.x.max()), float(surf_hs.y.min()), float(surf_hs.y.max())]

# grey land + hillshade
grey = mcolors.to_rgba('#dfdfdf')
base_rgba = np.zeros((shade.shape[0], shade.shape[1], 4), dtype=np.float32)
base_rgba[..., 0] = grey[0]
base_rgba[..., 1] = grey[1]
base_rgba[..., 2] = grey[2]
base_rgba[..., 3] = np.where(land, 1.0, 0.0)
ax.imshow(base_rgba, extent=sh_ext, transform=PROJ_3413, origin='upper', zorder=0.5)

shade_rgba = np.zeros((shade.shape[0], shade.shape[1], 4), dtype=np.float32)
shade_rgba[..., 0] = shade
shade_rgba[..., 1] = shade
shade_rgba[..., 2] = shade
shade_rgba[..., 3] = np.where(land, 0.60, 0.0)
ax.imshow(shade_rgba, extent=sh_ext, transform=PROJ_3413, origin='upper', zorder=1)

# white see-through fill over the ice
mask_clip = yslice(mask_da.sel(x=slice(xmin, xmax)), ymin, ymax)
ice_vals = xr.where(mask_clip > 0.5, 1.0, np.nan)
ax.pcolormesh(ice_vals.x, ice_vals.y, ice_vals, cmap=mcolors.ListedColormap(['white']), vmin=0, vmax=2,
              transform=PROJ_3413, alpha=0.75, zorder=2)

# no slowdown / no data
anom_clip_raw = yslice(anom_da.sel(x=slice(xmin, xmax)), ymin, ymax)
valid_clip = yslice(valid_mask_da.sel(x=slice(xmin, xmax)), ymin, ymax)
mask_aligned = mask_clip.reindex_like(anom_clip_raw, method='nearest')
valid_aligned = valid_clip.reindex_like(anom_clip_raw, method='nearest')

no_slowdown = xr.where((mask_aligned > 0.5) & valid_aligned & ~(anom_clip_raw < 0), 1.0, np.nan)
ax.pcolormesh(no_slowdown.x, no_slowdown.y, no_slowdown, cmap=mcolors.ListedColormap(['#BDB8B2']),
              vmin=0, vmax=2, transform=PROJ_3413, alpha=0.60, zorder=2.5)

no_data = xr.where((mask_aligned > 0.5) & (~valid_aligned), 1.0, 0.0)
with matplotlib.rc_context({'hatch.color': '#666666', 'hatch.linewidth': 0.4}):
    ax.contourf(no_data.x, no_data.y, no_data, levels=[0.5, 1.5], colors=['none'], hatches=['//'],
                transform=PROJ_3413, zorder=2.7)

# the W1 anomaly, slowdown only
anom_clip = anom_clip_raw.where(anom_clip_raw < 0)
im = ax.pcolormesh(anom_clip.x, anom_clip.y, anom_clip, cmap=CMAP_ANOM, norm=NORM_ANOM,
                   transform=PROJ_3413, alpha=0.82, zorder=3)
print('W1 anomaly min', float(np.nanmin(anom_clip.values)), 'mean', float(np.nanmean(anom_clip.values)))

# stippling where recovery took more than 1 winter
rec_clip = yslice(recovery_da.sel(x=slice(xmin, xmax)), ymin, ymax)
multi_yr = rec_clip.where((rec_clip > 1) & (rec_clip < 100))
rec_sub = multi_yr[::11, ::11]
xx, yy = np.meshgrid(rec_sub.x.values, rec_sub.y.values)
stip = np.isfinite(rec_sub.values)
ax.scatter(xx[stip], yy[stip], s=75, c='black', marker='.', alpha=0.99, linewidths=0, zorder=9,
           transform=PROJ_3413)

# ice sheet outline
ax.contour(mask_clip.x, mask_clip.y, mask_clip, levels=[0.5], colors='black', linewidths=1.0,
           transform=PROJ_3413, zorder=6)

# coastline
surf_clip = yslice(surf_da.sel(x=slice(xmin, xmax)), ymin, ymax)
ax.contour(surf_clip.x, surf_clip.y, (surf_clip <= 1.0).astype(float), levels=[0.5], colors='#3e3e3e',
           linewidths=1.2, transform=PROJ_3413, zorder=5)

# elevation contours
mask_on_surf = mask_clip.interp_like(surf_clip, method='nearest')
surf_ice = surf_clip.where(mask_on_surf > 0.5)
cs_elev = ax.contour(surf_ice.x, surf_ice.y, surf_ice, levels=list(range(200, 3400, 200)),
                     colors='black', linewidths=0.85, transform=PROJ_3413, zorder=7)
contour_labels = ax.clabel(cs_elev, levels=list(range(400, 3400, 400)), inline=True, inline_spacing=5,
                           fmt='%d', fontsize=FS_ELEV, use_clabeltext=True)
for txt in contour_labels:
    txt.set_fontweight('bold')
    txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])
    txt.set_clip_on(True)

# ROIs and labels
panel_w = xmax - xmin
panel_h = ymax - ymin
for i, row in rois.iterrows():
    geom = row.geometry
    cx = geom.centroid.x
    cy = geom.centroid.y
    roi_x, roi_y = geom.exterior.xy
    ax.plot(roi_x, roi_y, color='crimson', linewidth=2.5, transform=PROJ_3413, zorder=8)

    tx = cx - panel_w * ROI_LABEL_FRAC
    ty = cy + panel_h * ROI_LABEL_DY_FRAC
    ax.plot([cx, tx], [cy, ty], color='crimson', linewidth=1.0, transform=PROJ_3413, zorder=8)

    rr = min(panel_w, panel_h) * ROI_LABEL_RADIUS_FRAC
    ax.add_patch(Circle((tx, ty), rr, transform=PROJ_3413, facecolor='white', edgecolor='crimson',
                        linewidth=1.0, alpha=0.88, zorder=9))
    ax.text(tx, ty, f"L{row['label']}", transform=PROJ_3413, fontsize=FS_MAP_ROI_LABEL,
            fontweight='bold', ha='center', va='center', zorder=10)

legend_elements = [Patch(facecolor='#BDB8B2', alpha=0.75, edgecolor='#888888', linewidth=0.5, label='No slowdown'),
                   Patch(facecolor='none', edgecolor='#444444', hatch='////', linewidth=0.8, label='No data')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=FS_ANNOT, framealpha=0.88,
          edgecolor='#888888', handlelength=1.8, handleheight=1.2)

# gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0, x_inline=False, y_inline=False, dms=False)
gl.top_labels = True
gl.bottom_labels = False
gl.left_labels = True
gl.right_labels = False
gl.xlocator = mticker.FixedLocator(range(-80, -10, 1))
gl.ylocator = mticker.FixedLocator([v / 2 for v in range(110, 180)])
gl.xlabel_style = dict(fontsize=FS_TICK - 6, color='black', rotation=0)
gl.ylabel_style = dict(fontsize=FS_TICK - 6, color='black', rotation=90)

# 100 km scale bar, bottom left
x0 = xmin + 0.05 * panel_w
y0 = ymin + 0.06 * panel_h
x1 = x0 + 100_000
tick_h = 0.012 * panel_h
stroke = [pe.withStroke(linewidth=4.5, foreground='white')]
ax.plot([x0, x1], [y0, y0], color='black', linewidth=2.5, transform=PROJ_3413, solid_capstyle='butt',
        zorder=20, path_effects=stroke)
ax.plot([x0, x0], [y0 - tick_h, y0 + tick_h], color='black', linewidth=2.5, transform=PROJ_3413,
        zorder=20, path_effects=stroke)
ax.plot([x1, x1], [y0 - tick_h, y0 + tick_h], color='black', linewidth=2.5, transform=PROJ_3413,
        zorder=20, path_effects=stroke)
ax.text((x0 + x1) / 2, y0 + tick_h + 0.01 * panel_h, '100 km', ha='center', va='bottom',
        fontsize=FS_ANNOT - 2, fontweight='bold', color='black', transform=PROJ_3413, zorder=21,
        path_effects=[pe.withStroke(linewidth=4, foreground='white')])

ax.text(0.02, 0.98, '(a)', transform=ax.transAxes, fontsize=FS_LETTER, fontweight='bold', va='top', ha='left',
        path_effects=[pe.withStroke(linewidth=3, foreground='white')])


# (b)-(k) timeseries
ts_gap_in = 0.16
ts_h_in = (maps_h_in - ts_gap_in * (n_ts - 1)) / n_ts
letters = 'bcdefghijk'

for i in range(n_ts):
    lbl = SELECTED_ROI_LABELS[i]
    type_label = f'L{lbl}'
    bottom_in = maps_h_in - (i + 1) * ts_h_in - i * ts_gap_in
    ax = fig.add_axes([TS_X0, FIG_Y0 + bottom_in / fig_h, TS_X1 - TS_X0, ts_h_in / fig_h])

    df = pd.read_csv(os.path.join(PREP_DIR, f'ROI_{lbl}_timeseries.csv'), parse_dates=['date'])
    df = df.sort_values('date')

    # baseline and post-event windows
    baseline_wins = []
    post_wins = []
    for start_s, end_s in manual_windows[lbl]:
        t0 = pd.Timestamp(start_s)
        t1 = pd.Timestamp(end_s)
        if t1 <= EVENT_DATE:
            baseline_wins.append((t0, t1))
        else:
            post_wins.append((t0, t1))

    # baseline = median of the baseline winter medians
    baseline_vals = []
    for t0, t1 in baseline_wins:
        sub = df.loc[(df['date'] >= t0) & (df['date'] <= t1), 'velocity_smooth']
        if not sub.empty:
            baseline_vals.append(float(sub.median()))
    bmed = float(np.median(baseline_vals))

    # save the baseline and winter medians
    records = [{'roi_label': lbl, 'roi_type_label': type_label, 'period': 'baseline',
                'winter_start': '', 'winter_end': '', 'winter_label': 'baseline',
                'median_velocity_smooth': bmed, 'n_windows': len(baseline_vals)}]
    for t0, t1 in post_wins:
        sub = df.loc[(df['date'] >= t0) & (df['date'] <= t1), 'velocity_smooth']
        if sub.empty:
            continue
        records.append({'roi_label': lbl, 'roi_type_label': type_label, 'period': 'winter',
                        'winter_start': t0.date().isoformat(), 'winter_end': t1.date().isoformat(),
                        'winter_label': f'{t0.year}/{str(t1.year)[-2:]}',
                        'median_velocity_smooth': float(sub.median()), 'n_windows': int(sub.shape[0])})
    winter_csv = os.path.join(WINTER_MEDIAN_CSV_DIR, f'ROI_{lbl}_winter_medians.csv')
    pd.DataFrame.from_records(records).to_csv(winter_csv, index=False)

    # velocity points and the smoothed line
    ax.errorbar(df['date'], df['velocity'], yerr=df['velocity_error'], fmt='o', color='#444444',
                markersize=2.5, alpha=0.70, elinewidth=0.4, zorder=2)
    ax.plot(df['date'], df['velocity_smooth'], color='dodgerblue', linewidth=2.2, zorder=3)
    ax.set_xlim(df['date'].min() - pd.Timedelta(days=30), df['date'].max() + pd.Timedelta(days=30))

    # baseline line across all the baseline winters
    t_first = min([t0 for t0, t1 in baseline_wins])
    t_last = max([t1 for t0, t1 in baseline_wins])
    ax.hlines(bmed, t_first, t_last, color='k', linewidth=1.4, alpha=0.85, zorder=4)

    # winter medians and % change
    for t0, t1 in post_wins:
        sub = df.loc[(df['date'] >= t0) & (df['date'] <= t1), 'velocity_smooth']
        if sub.empty:
            continue
        lmed = sub.median()
        ax.hlines(lmed, t0, t1, color='crimson', linewidth=2.0, alpha=0.75, zorder=5)
        annotate_pct(ax, (lmed - bmed) / bmed * 100.0, t1, lmed)

    ax.axvline(EVENT_DATE, color='r', alpha=0.9, linestyle='--', linewidth=1.8, zorder=4)
    ax.set_title(f'({letters[i]}) {type_label}', loc='left', fontsize=25, fontweight='bold', y=0.8)

    # y axis on alternate sides
    if i % 2 == 0:
        ax.yaxis.set_label_position('left')
        ax.yaxis.tick_left()
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(False)
    else:
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(labelsize=FS_TICK)

    # y limits
    yvals = df['velocity'].to_numpy(dtype=float)
    yvals = yvals[np.isfinite(yvals)]
    ylo, yhi = np.percentile(yvals, [0, 99])
    span = max(float(yhi - ylo), 1.0)
    ylo = float(ylo - TS_YLIM_LOWER_PAD_FRAC * span)
    yhi = float(yhi + TS_YLIM_UPPER_PAD_FRAC * span)
    ax.set_ylim(ylo, yhi)

    # y ticks
    tick_step = 25
    for cand in [200, 175, 150, 125, 100, 75, 50, 25]:
        first_tick = np.ceil(ylo / cand) * cand
        last_tick = np.floor(yhi / cand) * cand
        if last_tick >= first_tick and int(np.floor((last_tick - first_tick) / cand) + 1) >= 3:
            tick_step = cand
            break
    tick_start = np.ceil(ylo / tick_step) * tick_step
    tick_end = np.floor(yhi / tick_step) * tick_step
    ticks = np.arange(tick_start, tick_end + 0.5 * tick_step, tick_step)
    if ticks.size < 3:
        ticks = np.array([tick_start, tick_start + tick_step, tick_start + 2 * tick_step])
        if ticks[-1] > yhi:
            ax.set_ylim(ylo, float(ticks[-1]))
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}'))

    ax.grid(True, alpha=0.7, linewidth=1.1, color='#888888', axis='x')
    if i < n_ts - 1:
        ax.set_xticklabels([])
    ax.set_ylabel('')

# shared y label to the right of the timeseries
fig.text(TS_X1 + 0.037, (FIG_Y0 + FIG_Y1) / 2, 'Ice speed (m yr$^{-1}$)', ha='center', va='center',
         fontsize=FS_LABEL, rotation=270)

# colourbar
map_w_fig = map_w_in / fig_w
cbar_w = map_w_fig * 0.88
cbar_x = map_x0_fig + (map_w_fig - cbar_w) / 2
cbar_ax = fig.add_axes([cbar_x, 0.08, cbar_w, 0.012])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', extend='min')
cbar.set_label('Winter 2021/22 velocity anomaly (%)', fontsize=FS_CBAR - 4, labelpad=5)
cbar.ax.tick_params(labelsize=FS_TICK - 6)

fig.text((TS_X0 + TS_X1) / 2, FIG_Y0 - 0.032, 'Year', ha='center', fontsize=FS_LABEL)

out = os.path.join(OUTPUT_DIR, 'Fig04_velocity_tseries.png')
plt.savefig(out, dpi=400, bbox_inches='tight')
