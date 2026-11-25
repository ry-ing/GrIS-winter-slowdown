# Maps and velocity timeseries for the marine-terminating glaciers M1-M3

import os
import sys
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
import cartopy.crs as ccrs
from matplotlib.colors import LightSource

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import config

PREP_DIR = os.path.join(config.DATA_DIR, 'velocity')
MTG_DIR = os.path.join(config.DATA_DIR, 'velocity_mtg')
OUTPUT_DIR = config.FIG_DIR

ROI_LABEL_FRAC = 0.2   # label offset to the left of the ROI, fraction of the map width

CMAP_ANOM = 'YlGnBu'
NORM_ANOM = mcolors.Normalize(vmin=-20, vmax=0)
OCEAN_COLOR = '#AFCED9'
EVENT_DATE = pd.to_datetime('2021-08-14')
PROJ_3413 = ccrs.Stereographic(central_latitude=90, central_longitude=-45, true_scale_latitude=70)

# layout, figure fractions
MAPS_X0 = 0.03
TS_X0 = 0.45
TS_X1 = 0.96
FIG_Y1 = 0.95

FS_TITLE = 20
FS_LABEL = 26
FS_TICK = 22
FS_ANNOT = 22
FS_CBAR = 24
FS_LETTER = 28
FS_ELEV = 14
SCALEBAR_LW = 3.5   # pt, stem width of DejaVu Sans Bold at FS_ANNOT - 2 (20 pt)

PCT_LABEL_DAYS_OFFSET = -7

# map sizes (inches) and zoom
MAP_W_IN = 4.5
MAP_H_IN = 4.5
GAP_IN = 0.4
M3_W_IN = MAP_W_IN * 2 + GAP_IN
M3_H_IN = 4.5
BASE_M_PER_INCH = 120_000 / MAP_W_IN
ZOOM = [1.0, 3.0, 1.4]   # M1, M2, M3

# fixed y ticks for each timeseries
Y_TICKS = {'M1': [100, 300, 500], 'M2': [100, 200, 300], 'M3': [100, 300, 500]}


# slice y
def yslice(da, ylo, yhi):
    if float(da.y[0]) > float(da.y[-1]):
        return da.sel(y=slice(yhi, ylo))
    return da.sel(y=slice(ylo, yhi))


# one map panel
def draw_map_panel(ax, geom, name, extent, zoom, anom_da, recovery_da):
    xmin, xmax, ymin, ymax = extent
    ax.set_extent([xmin, xmax, ymin, ymax], crs=PROJ_3413)
    ax.set_facecolor(OCEAN_COLOR)

    # grey land with hillshade
    surf_hs = yslice(surf_da.sel(x=slice(xmin - 5000, xmax + 5000)), ymin - 5000, ymax + 5000)
    surf_vals = surf_hs.values
    land = np.where(np.isfinite(surf_vals), surf_vals > 1.0, False)
    shade = LightSource(azdeg=315, altdeg=45).hillshade(np.where(land, surf_vals, 0.0), vert_exag=10, dx=150, dy=150)
    sh_ext = [float(surf_hs.x.min()), float(surf_hs.x.max()), float(surf_hs.y.min()), float(surf_hs.y.max())]
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

    # W1 anomaly, slowdown only
    anom_clip = yslice(anom_da.sel(x=slice(xmin, xmax)), ymin, ymax)
    anom_clip = anom_clip.where(anom_clip < 0)
    im = ax.pcolormesh(anom_clip.x, anom_clip.y, anom_clip, cmap=CMAP_ANOM, norm=NORM_ANOM,
                       transform=PROJ_3413, alpha=0.82, zorder=3)

    # stippling where recovery took more than 1 winter
    rec_clip = yslice(recovery_da.sel(x=slice(xmin, xmax)), ymin, ymax)
    multi_yr = rec_clip.where((rec_clip > 1) & (rec_clip < 100))
    stride = max(1, int(round(12 / zoom)))
    rec_sub = multi_yr[::stride, ::stride]
    xx, yy = np.meshgrid(rec_sub.x.values, rec_sub.y.values)
    stip = np.isfinite(rec_sub.values)
    ax.scatter(xx[stip], yy[stip], s=35, c='black', marker='.', alpha=0.75, linewidths=0, zorder=4,
               transform=PROJ_3413)

    # margin, coast and contours
    ax.contour(mask_clip.x, mask_clip.y, mask_clip, levels=[0.5], colors='black', linewidths=1.0,
               transform=PROJ_3413, zorder=6)
    surf_clip = yslice(surf_da.sel(x=slice(xmin, xmax)), ymin, ymax)
    ax.contour(surf_clip.x, surf_clip.y, (surf_clip <= 1.0).astype(float), levels=[0.5], colors='#3e3e3e',
               linewidths=1.2, transform=PROJ_3413, zorder=5)
    surf_ice = surf_clip.where(mask_clip.interp_like(surf_clip, method='nearest') > 0.5)
    cs = ax.contour(surf_ice.x, surf_ice.y, surf_ice, levels=list(range(200, 3400, 200)), colors='black',
                    linewidths=0.55, transform=PROJ_3413, zorder=7)
    contour_labels = ax.clabel(cs, levels=list(range(400, 3400, 400)), inline=True, inline_spacing=5,
                               fmt='%d', fontsize=FS_ELEV)
    for txt in contour_labels:
        txt.set_fontweight('bold')
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])

    # ROI and label
    cx = geom.centroid.x
    cy = geom.centroid.y
    roi_x, roi_y = geom.exterior.xy
    ax.plot(roi_x, roi_y, color='crimson', linewidth=2.5, transform=PROJ_3413, zorder=8)
    tx = cx - (xmax - xmin) * ROI_LABEL_FRAC
    ax.plot([cx, tx], [cy, cy], color='crimson', linewidth=1.0, transform=PROJ_3413, zorder=8)
    ax.text(tx, cy, name, transform=PROJ_3413, fontsize=FS_ANNOT, fontweight='bold', ha='center', va='center',
            zorder=9, bbox=dict(facecolor='white', alpha=0.88, edgecolor='crimson', boxstyle='circle,pad=0.2',
                                linewidth=1.0))

    # 10 km scale bar, SCALEBAR_LW matches the stroke of the bold label
    scalebar_len = 10_000
    scalebar_label = '10 km'

    panel_w = xmax - xmin
    panel_h = ymax - ymin
    sb_x = xmin + 0.05 * panel_w
    sb_y = ymin + 0.06 * panel_h
    if name == 'M1':   # centred, the label runs off the panel edge otherwise
        sb_x = (xmin + xmax) / 2 - scalebar_len / 2
    tick_h = 0.012 * panel_h
    stroke = [pe.withStroke(linewidth=SCALEBAR_LW + 2.5, foreground='white')]
    xs = [sb_x, sb_x + scalebar_len]
    for path in [(xs, [sb_y, sb_y]), ([xs[0], xs[0]], [sb_y - tick_h, sb_y + tick_h]),
                 ([xs[1], xs[1]], [sb_y - tick_h, sb_y + tick_h])]:
        ax.plot(path[0], path[1], color='black', linewidth=SCALEBAR_LW, transform=PROJ_3413, zorder=9,
                solid_capstyle='butt', path_effects=stroke)
    # redraw the bar without the outline so the tick outlines do not cut into it
    ax.plot(xs, [sb_y, sb_y], color='black', linewidth=SCALEBAR_LW, transform=PROJ_3413, zorder=9.1,
            solid_capstyle='butt')
    ax.text(sb_x + scalebar_len / 2, sb_y + panel_h * 0.02, scalebar_label, transform=PROJ_3413,
            fontsize=FS_ANNOT - 2, fontweight='bold', ha='center', va='bottom', zorder=9)

    gl = ax.gridlines(draw_labels=True, linewidth=0, x_inline=False, y_inline=False)
    gl.top_labels = True
    gl.bottom_labels = False
    gl.left_labels = True
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(range(-80, -10, 1))
    gl.ylocator = mticker.FixedLocator([v / 2 for v in range(110, 180)])
    gl.xlabel_style = dict(fontsize=FS_TICK - 6, color='black', rotation=0)
    gl.ylabel_style = dict(fontsize=FS_TICK - 6, color='black', rotation=90)
    return im


os.makedirs(OUTPUT_DIR, exist_ok=True)

mask_da = xr.open_dataarray(os.path.join(config.CACHE_DIR, 'ice_mask.nc'))
surf_da = xr.open_dataarray(os.path.join(config.CACHE_DIR, 'bed_surface.nc'))

# the marine ROIs, north to south = M1, M2, M3
rois = gpd.read_file(config.ROI_SHAPEFILE_PATH).to_crs('EPSG:3413')
rois['id'] = rois['id'].astype(int)
rois = rois[rois['id'].isin(config.MTG_ROI_LABELS)].copy()
rois['cy'] = rois.geometry.centroid.y
rois = rois.sort_values('cy', ascending=False).reset_index(drop=True)
names = ['M1', 'M2', 'M3']

fig = plt.figure(figsize=(26.0, 14.0))
fig_w = 26.0
fig_h = 14.0

# map extents
extents = []
panel_sizes = [(MAP_W_IN, MAP_H_IN), (MAP_W_IN, MAP_H_IN), (M3_W_IN, M3_H_IN)]
for i in range(3):
    b = rois.iloc[i].geometry.bounds
    cx = (b[0] + b[2]) / 2
    cy = (b[1] + b[3]) / 2
    w_m = panel_sizes[i][0] * BASE_M_PER_INCH / ZOOM[i]
    h_m = panel_sizes[i][1] * BASE_M_PER_INCH / ZOOM[i]
    extents.append((cx - w_m / 2, cx + w_m / 2, cy - h_m / 2, cy + h_m / 2))

w_frac = MAP_W_IN / fig_w
h_frac = MAP_H_IN / fig_h
gap_w_frac = GAP_IN / fig_w
gap_h_frac = GAP_IN / fig_h
w3_frac = M3_W_IN / fig_w
h3_frac = M3_H_IN / fig_h
y_top = FIG_Y1 - h_frac
y_m3 = y_top - gap_h_frac - h3_frac

map_positions = [[MAPS_X0, y_top, w_frac, h_frac],
                 [MAPS_X0 + w_frac + gap_w_frac, y_top, w_frac, h_frac],
                 [MAPS_X0, y_m3, w3_frac, h3_frac]]

for i in range(3):
    ax = fig.add_axes(map_positions[i], projection=PROJ_3413)
    anom_da = xr.open_dataarray(os.path.join(MTG_DIR, f'spatial_anomaly_w1_{names[i]}.nc'))
    recovery_da = xr.open_dataarray(os.path.join(MTG_DIR, f'recovery_time_{names[i]}.nc'))
    im = draw_map_panel(ax, rois.iloc[i].geometry, names[i], extents[i], ZOOM[i], anom_da, recovery_da)
    if i == 0:
        im_ref = im
    ax.text(0.02, 0.98, f'({"abc"[i]})', transform=ax.transAxes, fontsize=FS_LETTER, fontweight='bold',
            va='top', ha='left', path_effects=[pe.withStroke(linewidth=3, foreground='white')])


# timeseries
ts_total_h = FIG_Y1 - y_m3
ts_gap = 0.05
ts_h = (ts_total_h - ts_gap * 2) / 3

for i in range(3):
    roi_id = int(rois.iloc[i]['id'])
    name = names[i]
    ax = fig.add_axes([TS_X0, FIG_Y1 - (i + 1) * ts_h - i * ts_gap, TS_X1 - TS_X0, ts_h])

    df = pd.read_csv(os.path.join(MTG_DIR, f'ROI_{roi_id}_timeseries.csv'), parse_dates=['date'])
    df = df.sort_values('date')
    ax.errorbar(df['date'], df['velocity'], yerr=df['velocity_error'], fmt='o', color='#444444', markersize=2.5,
                alpha=0.70, elinewidth=0.4, zorder=2)
    ax.plot(df['date'], df['velocity_smooth'], color='dodgerblue', linewidth=2.2, zorder=3)

    # Nov-Apr winters
    df['season_year'] = np.where(df['date'].dt.month >= 11, df['date'].dt.year + 1, df['date'].dt.year)
    df_win = df[df['date'].dt.month.isin([11, 12, 1, 2, 3, 4])]
    bmed = df_win.loc[df_win['season_year'] <= 2021, 'velocity_smooth'].median()
    ax.hlines(bmed, df['date'].min(), pd.Timestamp('2021-04-30'), color='k', linewidth=1.4, alpha=0.85, zorder=4)

    for season, grp in df_win.groupby('season_year'):
        if season < 2022:
            continue
        lstart = pd.Timestamp(f'{season - 1}-11-01')
        lend = pd.Timestamp(f'{season}-04-30')
        lmed = grp['velocity_smooth'].median()
        pct = (lmed - bmed) / bmed * 100.0
        ax.hlines(lmed, lstart, lend, color='crimson', linewidth=2.0, alpha=0.75, zorder=5)
        if pct < 0:
            color = 'darkred'
        else:
            color = 'darkgreen'
        ann = ax.annotate(f'{pct:+.1f}%', xy=(lend + pd.Timedelta(days=PCT_LABEL_DAYS_OFFSET), lmed),
                          xytext=(0, -18), textcoords='offset points', color=color, fontsize=FS_ANNOT - 2,
                          va='top', ha='left', zorder=10,
                          bbox=dict(facecolor='white', alpha=0.45, edgecolor=color, boxstyle='round,pad=0.15',
                                    linewidth=0.5))
        ann.get_bbox_patch().set_clip_on(False)
        print(f'{name} {season - 1}/{str(season)[-2:]}: {pct:+.1f}%')

    ax.axvline(EVENT_DATE, color='k', alpha=0.65, linestyle='--', linewidth=1.4, zorder=4)
    ax.set_title(f'({"def"[i]}) {name}', loc='left', fontsize=FS_TITLE, fontweight='bold', y=0.88)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.45, linewidth=0.7, color='#888888')
    if i < 2:
        ax.set_xticklabels([])
    ax.yaxis.set_major_locator(mticker.FixedLocator(Y_TICKS[name]))
    ax.set_ylim(top=Y_TICKS[name][-1])

fig.text(TS_X0 - 0.04, y_m3 + ts_total_h / 2, 'Ice speed (m yr$^{-1}$)', fontsize=FS_LABEL, rotation=90,
         ha='center', va='center')
fig.text(TS_X0 + (TS_X1 - TS_X0) / 2, y_m3 - 0.05, 'Year', fontsize=FS_LABEL, ha='center')

# colourbar under the maps
cbar_w = w3_frac * 0.88
cbar_ax = fig.add_axes([MAPS_X0 + (w3_frac - cbar_w) / 2, y_m3 - 0.03, cbar_w, 0.018])
cbar = fig.colorbar(im_ref, cax=cbar_ax, orientation='horizontal', extend='min')
cbar.set_label('Winter 2021/22 velocity anomaly (%)', fontsize=FS_CBAR - 4, labelpad=5)
cbar.ax.tick_params(labelsize=FS_TICK - 6)

out = os.path.join(OUTPUT_DIR, 'Fig10_mtg_velocity_tseries.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
