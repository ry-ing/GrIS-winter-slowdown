# Aug 2021 melt event, plus an extra fig of runoff by elevation

import os
import sys
import socket
import numpy as np
import xarray as xr
import rioxarray
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cmocean
from matplotlib.colors import LightSource
from scipy.stats import gaussian_kde
from scipy.ndimage import distance_transform_edt
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import warnings

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import config

warnings.filterwarnings("ignore")

MAR_FILE = os.path.join(config.MAR_DIR, 'MARv3.14.3-10km-daily-ERA5-{year}.nc')
BEDMACHINE_PATH = config.BEDMACHINE_PATH
MASK_PATH_FILE = config.MASK_PATH_FILE
ROI_SHAPEFILE_PATH = config.ROI_SHAPEFILE_PATH
OUTPUT_DIR = config.FIG_DIR
CACHE_DIR = config.CACHE_DIR

# MAR region (EPSG:3413, m)
MIN_X = -340000.0
MIN_Y = -2750000.0
MAX_X = 50000.0
MAX_Y = -2000000.0
MAR_REGION = {'minx': MIN_X, 'miny': MIN_Y, 'maxx': MAX_X, 'maxy': MAX_Y}
PROJ_3413 = ccrs.Stereographic(central_latitude=90, central_longitude=-45, true_scale_latitude=70)

BASELINE_YEARS = range(1950, 2021)
ALL_YEARS = range(1950, 2026)
TYPICAL_YEAR = 2018
PIXEL_AREA_M2 = (10e3) ** 2
N_DAYS_JJA = 92

# map x limits, the y limits come from the ROIs
MAP_XMIN = -340000
MAP_XMAX = -50000
ROI_EXTENT_PAD_M = 40_000.0
SELECTED_ROI_LABELS = config.ROI_LABELS
PANEL_A_VMAX_CM_DAY = 11.0

# elevation bands: min, max, colour, label
ELEV_BANDS = [
    (0,    500,  '#C6DBEF', '0–500 m'),
    (500,  1000, '#6BAED6', '500–1000 m'),
    (1000, 1500, '#2171B5', '1000–1500 m'),
    (1500, 2000, '#08306B', '1500-2000 m'),
]

# years to circle in (c) and label positions
SCATTER_HIGHLIGHT_YEARS = {2012, 2019, 2022}
LABEL_ANCHORS = {
    '14th August 2021': (0.65, 0.95),
    '3rd September 2022': (0.42, 0.85),
    '28th July 2012': (0.7, 0.7),
    '31st July 2019': (0.81, 0.30),
    '11th August 1997': (0.25, 0.72),
    '18th July 2002': (0.35, 0.75),
}

# caches
REFRESH_PANEL_A = False
REFRESH_PANEL_B = False
REFRESH_PANEL_C = False
PANEL_A_CACHE = os.path.join(CACHE_DIR, 'panel_a_runoff_change_20210814_vs_pre3.nc')
PANEL_B_CACHE = os.path.join(CACHE_DIR, 'panel_b_elevation_bands_2021_jja.nc')
PANEL_B_TYPICAL_CACHE = os.path.join(CACHE_DIR, f'panel_b_elevation_bands_{TYPICAL_YEAR}_jja.nc')
PANEL_C_CACHE = os.path.join(CACHE_DIR, 'panel_c_scatter_1950_2025_jjas.npz')

SLURM_CONFIG = {
    "account": "geos_extra",
    "cores": 8,
    "memory": "32GB",
    "queue": "General",
    "walltime": "04:00:00",
    "log_directory": '/home/s1834371/scratch/dask_log',
    "job_extra_directives": ["--job-name=dask-melt-fig", "--output=/dev/null", "--error=/dev/null",
                             f"--nodelist={socket.gethostname().split('.')[0]}"],
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

cluster = SLURMCluster(**SLURM_CONFIG)
cluster.scale(jobs=4)
client = Client(cluster)


# MAR runoff for one year
def open_ru(year):
    ds = xr.open_dataset(MAR_FILE.format(year=year), chunks={'TIME': -1})
    ru = ds['RU'].rename({'TIME': 'time'})
    x_m = np.round(ru.x.values * 1000).astype(np.float64)
    y_m = np.round(ru.y.values * 1000).astype(np.float64)
    ru = ru.assign_coords(x=x_m, y=y_m)
    ru = ru.rio.set_spatial_dims(x_dim='x', y_dim='y')
    ru = ru.rio.write_crs("EPSG:3413")
    ru = ru.rio.clip_box(**MAR_REGION)
    return ru.sum(dim='SECTOR')


# JJA runoff in each elevation band
def band_runoff(year):
    ru = open_ru(year).where(mask_on_mar == 1)
    ru = ru.sel(time=slice(f'{year}-06-01', f'{year}-08-31'))
    out = []
    for band in ELEV_BANDS:
        emin = band[0]
        emax = band[1]
        ru_band = ru.where((surface_on_mar >= emin) & (surface_on_mar < emax))
        total = ru_band.sum(dim=['x', 'y']) * PIXEL_AREA_M2 * 1e-3 * 1000 / 1e12
        out.append(total.compute().values)
    return np.array(out)


# band runoff, cached
def cached_band_runoff(year, path, refresh):
    if os.path.exists(path) and not refresh:
        with xr.open_dataset(path) as c:
            return c['band_vals'].values.astype(float)

    vals = band_runoff(year).astype(float)
    ds = xr.Dataset({'band_vals': (('band', 'day'), vals)})
    ds = ds.assign_coords(band=np.arange(len(ELEV_BANDS), dtype=np.int32))
    ds = ds.assign_coords(day=np.arange(N_DAYS_JJA, dtype=np.int32))
    ds = ds.assign_coords(elev_min=('band', np.array([b[0] for b in ELEV_BANDS], dtype=np.int32)))
    ds = ds.assign_coords(elev_max=('band', np.array([b[1] for b in ELEV_BANDS], dtype=np.int32)))
    ds = ds.assign_coords(label=('band', np.array([b[3] for b in ELEV_BANDS], dtype='U32')))
    ds.to_netcdf(path)
    return vals


# stacked bars
def stacked_bars(ax, vals, lw):
    bottom = np.zeros(N_DAYS_JJA)
    for band, v in zip(ELEV_BANDS, vals):
        ax.bar(np.arange(N_DAYS_JJA), v, bottom=bottom, color=band[2], edgecolor='black', linewidth=lw,
               width=1.0, align='center', label=band[3], zorder=2)
        bottom += v
    ax.set_xlim(-0.5, N_DAYS_JJA - 0.5)
    ax.set_ylim(bottom=0)
    ax.set_xticks([0, 30, 61])   # 1 Jun, 1 Jul, 1 Aug
    ax.set_xticklabels(['Jun', 'Jul', 'Aug'])
    ax.set_xticks(np.arange(0, N_DAYS_JJA), minor=True)
    ax.grid(True, axis='y', alpha=0.25, linewidth=0.5)
    ax.spines[['top', 'right']].set_visible(False)


# date as e.g. 11th August 1997
def fmt_date(t):
    t = pd.Timestamp(t)
    if 10 <= t.day % 100 <= 20:
        suffix = 'th'
    elif t.day % 10 == 1:
        suffix = 'st'
    elif t.day % 10 == 2:
        suffix = 'nd'
    elif t.day % 10 == 3:
        suffix = 'rd'
    else:
        suffix = 'th'
    return f'{t.day}{suffix} {t.strftime("%B %Y")}'


# map north-south extent from the ROIs
rois = gpd.read_file(ROI_SHAPEFILE_PATH).to_crs('EPSG:3413')
rois = rois[rois['id'].astype(int).isin(SELECTED_ROI_LABELS)]
bounds = rois.total_bounds   # minx, miny, maxx, maxy
MAP_YMIN = float(bounds[1] - ROI_EXTENT_PAD_M)
MAP_YMAX = float(bounds[3] + ROI_EXTENT_PAD_M)


# reference grids
bed_ds = xr.open_dataset(BEDMACHINE_PATH)
bed_ds = bed_ds.sel(x=slice(MAP_XMIN - 10000, MAP_XMAX + 10000),
                    y=slice(MAP_YMAX + 10000, MAP_YMIN - 10000))
surface_bedm = bed_ds['surface']

mask_ds = xr.open_dataset(MASK_PATH_FILE).rio.write_crs("EPSG:3413")
mask_ds = mask_ds.rio.clip_box(minx=MIN_X - 20000, miny=MIN_Y - 20000, maxx=MAX_X + 20000, maxy=MAX_Y + 20000)
mask_hires = xr.where(mask_ds['Band1'] == 2, 1, 0)
mask_on_bedm = mask_hires.interp_like(surface_bedm, method='nearest').fillna(0)

# MAR ice mask (MSK > 50) and surface height
ref = open_ru(2021)
static = xr.open_dataset(MAR_FILE.format(year=2021))[['MSK', 'SH']]
static = static.assign_coords(x=np.round(static.x.values * 1000).astype(np.float64),
                              y=np.round(static.y.values * 1000).astype(np.float64))
static = static.rio.set_spatial_dims(x_dim='x', y_dim='y')
static = static.rio.write_crs("EPSG:3413")
static = static.rio.clip_box(**MAR_REGION)
mask_on_mar = (static['MSK'] > 50).astype(float).interp_like(ref.isel(time=0), method='nearest').fillna(0)
surface_on_mar = static['SH']


# (a) runoff change on 14 Aug
if os.path.exists(PANEL_A_CACHE) and not REFRESH_PANEL_A:
    with xr.open_dataset(PANEL_A_CACHE) as c:
        runoff_change_plot = c['runoff_change_plot'].load()
else:
    ru_2021 = open_ru(2021).where(mask_on_mar == 1)
    pre3 = ru_2021.sel(time=slice('2021-08-11', '2021-08-13')).mean(dim='time').compute()
    aug14 = ru_2021.sel(time='2021-08-14').squeeze().compute()
    delta = (aug14 - pre3).where(mask_on_mar == 1)

    # onto BedMachine grid, fill to the margin
    delta_on_bed = delta.interp_like(surface_bedm, method='nearest')
    vals = delta_on_bed.values.astype(np.float64)
    ice = mask_on_bedm.values == 1
    valid = ice & np.isfinite(vals)
    dist, nearest = distance_transform_edt(~valid, return_indices=True)
    ny = nearest[0]
    nx = nearest[1]
    fill = ice & ~valid
    vals[fill] = vals[ny[fill], nx[fill]]
    runoff_change_plot = xr.DataArray(vals, coords=delta_on_bed.coords, dims=delta_on_bed.dims)
    runoff_change_plot = runoff_change_plot.where(ice).rename('runoff_change_plot')
    runoff_change_plot.to_netcdf(PANEL_A_CACHE)


# (b) elevation bands
band_vals_2021 = cached_band_runoff(2021, PANEL_B_CACHE, REFRESH_PANEL_B)
band_vals_typical = cached_band_runoff(TYPICAL_YEAR, PANEL_B_TYPICAL_CACHE, REFRESH_PANEL_B)
event_day_x = (pd.Timestamp('2021-08-14') - pd.Timestamp('2021-06-01')).days


# (c) runoff vs 3-day increase, Jun-Sep 1950-2025
if os.path.exists(PANEL_C_CACHE) and not REFRESH_PANEL_C:
    c = np.load(PANEL_C_CACHE)
    scatter_runoff = c['scatter_runoff'].astype(float)
    scatter_delta = c['scatter_delta'].astype(float)
    scatter_dates = c['scatter_dates_ns'].astype('datetime64[ns]')
    aug14_x = float(c['scatter_aug14_x'][0])
    aug14_y = float(c['scatter_aug14_y'][0])
    highlight_days = {}
    for i, yr in enumerate(c['highlight_years'].astype(int)):
        highlight_days[int(yr)] = {'runoff': float(c['highlight_runoff'][i]),
                                   'delta': float(c['highlight_delta'][i]),
                                   'date': pd.Timestamp(c['highlight_dates_ns'][i].astype('datetime64[ns]'))}
else:
    all_runoff = []
    all_delta = []
    all_dates = []
    highlight_days = {}   # peak runoff day with a positive increase in each highlight year
    for yr in ALL_YEARS:
        ru = open_ru(yr).where(mask_on_mar == 1).sel(time=slice(f'{yr}-06-01', f'{yr}-09-30'))
        ts = (ru.sum(dim=['x', 'y']) * PIXEL_AREA_M2 * 1e-3 * 1000 / 1e12).compute()
        vals = ts.values.astype(float)
        times = pd.DatetimeIndex(ts.time.values)
        # 3-day increase
        for i in range(3, len(vals)):
            d = vals[i] - np.mean(vals[i - 3:i])
            t = pd.Timestamp(times[i]).normalize()
            all_runoff.append(vals[i])
            all_delta.append(d)
            all_dates.append(t)
            if yr == 2021 and t.date() == pd.Timestamp('2021-08-14').date():
                aug14_x = vals[i]
                aug14_y = d
            if yr in SCATTER_HIGHLIGHT_YEARS and d > 0:
                if yr not in highlight_days or vals[i] > highlight_days[yr]['runoff']:
                    highlight_days[yr] = {'runoff': float(vals[i]), 'delta': float(d), 'date': t}

    all_delta = np.array(all_delta)
    pos = all_delta > 0
    scatter_runoff = np.array(all_runoff)[pos]
    scatter_delta = all_delta[pos]
    scatter_dates = np.array(all_dates).astype('datetime64[ns]')[pos]

    hl_years = np.array(sorted(highlight_days), dtype=np.int32)
    hl_runoff = np.array([highlight_days[y]['runoff'] for y in hl_years], dtype=np.float64)
    hl_delta = np.array([highlight_days[y]['delta'] for y in hl_years], dtype=np.float64)
    hl_dates = np.array([np.datetime64(highlight_days[y]['date'], 'ns').astype('int64') for y in hl_years],
                        dtype=np.int64)
    np.savez(PANEL_C_CACHE,
             scatter_runoff=scatter_runoff,
             scatter_delta=scatter_delta,
             scatter_dates_ns=scatter_dates.astype('int64'),
             scatter_aug14_x=np.array([aug14_x], dtype=np.float64),
             scatter_aug14_y=np.array([aug14_y], dtype=np.float64),
             highlight_years=hl_years,
             highlight_runoff=hl_runoff,
             highlight_delta=hl_delta,
             highlight_dates_ns=hl_dates,
             total_days_count=np.array([len(pos)], dtype=np.int64))

print('14 Aug 2021 runoff', round(aug14_x, 3), 'increase', round(aug14_y, 3), 'Gt/d')

# points to circle
highlights = [(aug14_x, aug14_y, pd.Timestamp('2021-08-14'))]
for yr in sorted(highlight_days):
    info = highlight_days[yr]
    highlights.append((info['runoff'], info['delta'], info['date']))
order = np.argsort(-scatter_delta)
for i in order[2:4]:
    highlights.append((scatter_runoff[i], scatter_delta[i], scatter_dates[i]))


# main figure
FONT = 28
FONT_SM = 24
FONT_PL = 34
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
fig = plt.figure(figsize=(26, 20))
fig.patch.set_facecolor('white')
ax_map = fig.add_axes([0.07, 0.07, 0.46, 0.89], projection=PROJ_3413)
ax_elev = fig.add_axes([0.60, 0.54, 0.36, 0.42])
ax_roc = fig.add_axes([0.60, 0.07, 0.36, 0.42])

band_legend = []
for band in ELEV_BANDS:
    band_legend.append(mpatches.Patch(facecolor=band[2], edgecolor='black', linewidth=0.5, label=band[3]))

# (a) map
ax_map.set_extent([MAP_XMIN, MAP_XMAX, MAP_YMIN, MAP_YMAX], crs=PROJ_3413)
ax_map.set_facecolor('#AED9E0')

# hillshade, transparent over the ocean
shade = LightSource(azdeg=315, altdeg=45).hillshade(bed_ds['bed'].fillna(0).values, vert_exag=8, dx=150, dy=150)
rgba = np.zeros((shade.shape[0], shade.shape[1], 4), dtype=np.float32)
rgba[..., 0] = shade
rgba[..., 1] = shade
rgba[..., 2] = shade
rgba[..., 3] = np.where(bed_ds['mask'].values != 0, 0.60, 0.0)
bed_extent = [float(bed_ds.x.min()), float(bed_ds.x.max()), float(bed_ds.y.min()), float(bed_ds.y.max())]
ax_map.imshow(rgba, extent=bed_extent, transform=PROJ_3413, origin='upper', zorder=1)

change_cm = runoff_change_plot / 10.0   # mm/day -> cm/day
vmin_plot = 0.0
vmax_plot = PANEL_A_VMAX_CM_DAY
im = ax_map.pcolormesh(change_cm.x, change_cm.y, change_cm, cmap='YlGnBu_r',
                       norm=mcolors.Normalize(vmin=vmin_plot, vmax=vmax_plot, clip=True),
                       transform=PROJ_3413, alpha=0.85, zorder=2)

# coastline, elevation contours on ice, ice margin
ax_map.contour(surface_bedm.x, surface_bedm.y, (surface_bedm <= 1.0).astype(float), levels=[0.5],
               colors='#444444', linewidths=1.2, transform=PROJ_3413, zorder=5)
surface_ice = surface_bedm.where(mask_on_bedm == 1)
cs = ax_map.contour(surface_ice.x, surface_ice.y, surface_ice, levels=[500, 1000, 1500, 2000],
                    colors='black', linewidths=1.6, linestyles='solid', transform=PROJ_3413, zorder=6)
contour_labels = ax_map.clabel(cs, levels=[500, 1000, 1500, 2000], inline=True, inline_spacing=5,
                               fmt='%d m', fontsize=FONT_SM, use_clabeltext=True)
for txt in contour_labels:
    txt.set_fontweight('bold')
    txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])
    txt.set_clip_on(True)
ax_map.contour(mask_hires.x, mask_hires.y, mask_hires, levels=[0.5], colors='black', linewidths=2.5,
               transform=PROJ_3413, zorder=8)

gl = ax_map.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False,
                      linewidth=0.4, color='gray', alpha=0.45, linestyle='--', zorder=3)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': FONT_SM, 'color': 'black', 'rotation': 0}
gl.ylabel_style = {'size': FONT_SM, 'color': 'black', 'rotation': 0}
ax_map.text(0.02, 0.98, '(a)', transform=ax_map.transAxes, fontsize=FONT_PL, fontweight='bold',
            va='top', color='black', zorder=25)

# 100 km scale bar, bottom left
sb_x0 = MAP_XMIN + 18_000
sb_y0 = MAP_YMIN + 40_000
tick_h = 8_000
sb_x1 = sb_x0 + 100_000
stroke = [pe.withStroke(linewidth=4.5, foreground='white')]
ax_map.plot([sb_x0, sb_x1], [sb_y0, sb_y0], color='black', linewidth=2.5, transform=PROJ_3413,
            solid_capstyle='butt', zorder=20, path_effects=stroke)
ax_map.plot([sb_x0, sb_x0], [sb_y0 - tick_h, sb_y0 + tick_h], color='black', linewidth=2.5,
            transform=PROJ_3413, zorder=20, path_effects=stroke)
ax_map.plot([sb_x1, sb_x1], [sb_y0 - tick_h, sb_y0 + tick_h], color='black', linewidth=2.5,
            transform=PROJ_3413, zorder=20, path_effects=stroke)
ax_map.text((sb_x0 + sb_x1) / 2, sb_y0 + tick_h + 4_000, '100 km', ha='center', va='bottom',
            fontsize=FONT_SM - 2, fontweight='bold', color='black', transform=PROJ_3413, zorder=21,
            path_effects=[pe.withStroke(linewidth=4, foreground='white')])

# (b) 2021 elevation bands
ax_elev.set_facecolor('white')
stacked_bars(ax_elev, band_vals_2021, lw=0.2)
ax_elev.axvline(event_day_x, color='black', linewidth=2.0, linestyle=':', zorder=5)
ax_elev.text(event_day_x + 1.0, ax_elev.get_ylim()[1] * 0.97, '14th August 2021', fontsize=FONT_SM,
             va='top', ha='left', style='italic')
ax_elev.tick_params(axis='x', which='minor', length=3, width=0.6, color='gray')
ax_elev.tick_params(axis='x', which='major', length=6)
ax_elev.set_ylabel('Total Runoff (Gt d$^{-1}$)')
ax_elev.legend(handles=band_legend, loc='upper left', bbox_to_anchor=(0.02, 0.90),
               framealpha=0.92, edgecolor='lightgray')
ax_elev.text(0.02, 0.97, '(b)', transform=ax_elev.transAxes, fontsize=FONT_PL, fontweight='bold', va='top')

# (c) scatter coloured by point density
ax_roc.set_facecolor('white')
xy = np.vstack([scatter_runoff, scatter_delta])
density = gaussian_kde(xy)(xy)
density_norm = (density - density.min()) / (density.max() - density.min())
sc = ax_roc.scatter(scatter_runoff, scatter_delta, s=80, alpha=0.7, linewidths=0.4, edgecolors='#444444',
                    zorder=2, c=density_norm, cmap=cmocean.cm.haline)

# circle and label
labels = {}
for x, y, t in highlights:
    ax_roc.scatter(x, y, s=360, facecolors='none', edgecolors='black', linewidths=2.3, zorder=6)
    if fmt_date(t) not in labels:
        labels[fmt_date(t)] = (float(x), float(y))

# label from highest to lowest increase
label_order = sorted(labels, key=lambda lab: labels[lab][1], reverse=True)
for label in label_order:
    x = labels[label][0]
    y = labels[label][1]
    tx = LABEL_ANCHORS[label][0]
    ty = LABEL_ANCHORS[label][1]
    if tx < 0.85:
        ha = 'left'
    else:
        ha = 'right'
    ax_roc.annotate(label, xy=(x, y), xycoords='data', xytext=(tx, ty), textcoords='axes fraction',
                    fontsize=FONT_SM - 3, color='black', style='italic', ha=ha,
                    va='center', arrowprops=dict(arrowstyle='->', color='black', lw=1.4),
                    bbox=dict(facecolor='white', alpha=0.90, edgecolor='none', pad=0.2), zorder=7)

ax_roc.axhline(0, color='gray', linewidth=0.8, linestyle='--', zorder=1)
ax_roc.set_xlabel('Total Daily Runoff (Gt)')
ax_roc.set_ylabel('Increase in runoff (Gt d$^{-1}$)')
ax_roc.grid(True, alpha=0.2, linewidth=0.5)
ax_roc.spines[['top', 'right']].set_visible(False)
ax_roc.text(0.02, 0.97, '(c)', transform=ax_roc.transAxes, fontsize=FONT_PL, fontweight='bold', va='top')

cbar_c = fig.colorbar(sc, cax=ax_roc.inset_axes([0.03, 0.52, 0.032, 0.30]), orientation='vertical')
cbar_c.set_ticks([0, 1])
cbar_c.set_ticklabels(['Low', 'High'], fontsize=FONT_SM - 6)
cbar_c.set_label('Density', fontsize=FONT_SM - 4, labelpad=4, rotation=270, va='bottom')
cbar_c.ax.tick_params(length=3, labelsize=FONT_SM - 6)

# line up (b) and (c) with the map
fig.draw_without_rendering()
map_pos = ax_map.get_position()
right_x0 = map_pos.x1 + 0.05
panel_h = (map_pos.height - 0.03) / 2.0
ax_elev.set_position([right_x0, map_pos.y0 + panel_h + 0.03, 0.97 - right_x0, panel_h])
ax_roc.set_position([right_x0, map_pos.y0, 0.97 - right_x0, panel_h])

cax = fig.add_axes([map_pos.x0, map_pos.y0 - 0.055, map_pos.width, 0.020])
cbar = fig.colorbar(im, cax=cax, orientation='horizontal', extend='max')
ticks = np.linspace(vmin_plot, vmax_plot, 6)
cbar.set_ticks(ticks)
cbar.set_ticklabels([f'{v:.2g}' for v in ticks])
cbar.set_label('Increase in runoff (cm day$^{-2}$)', fontsize=FONT, labelpad=4)
cbar.ax.tick_params(labelsize=FONT_SM - 2, length=4)

out_path = os.path.join(OUTPUT_DIR, 'Fig03_melt-event.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')


# extra figure, 2021 vs a typical year and the 1950-2020 mean
totals_2021 = np.nansum(band_vals_2021, axis=1)
totals_typical = np.nansum(band_vals_typical, axis=1)
share_2021 = totals_2021 / np.nansum(totals_2021)
share_typical = totals_typical / np.nansum(totals_typical)

baseline_shares = []
for yr in BASELINE_YEARS:
    totals = np.nansum(band_runoff(yr), axis=1).astype(float)
    baseline_shares.append(totals / np.nansum(totals))
share_baseline = np.nanmean(np.array(baseline_shares), axis=0)

SUP_FONT = 11
SUP_FONT_SM = 9
SUP_FONT_PL = 13
fig_supp = plt.figure(figsize=(11.2, 13.6))
fig_supp.patch.set_facecolor('white')
gs = fig_supp.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.85], hspace=0.35)
ax_s1 = fig_supp.add_subplot(gs[0])
ax_s2 = fig_supp.add_subplot(gs[1])
ax_s3 = fig_supp.add_subplot(gs[2])

# (a) 2021, same as main panel (b)
ax_s1.set_facecolor('white')
stacked_bars(ax_s1, band_vals_2021, lw=0.15)
ax_s1.axvline(event_day_x, color='black', linewidth=1.4, linestyle=':', zorder=5)
ax_s1.text(event_day_x + 1.0, ax_s1.get_ylim()[1] * 0.97, '14th August 2021', fontsize=SUP_FONT_SM,
           va='top', ha='left', style='italic')
ax_s1.tick_params(axis='x', which='minor', length=2, width=0.5, color='gray')
ax_s1.tick_params(axis='both', which='major', labelsize=SUP_FONT_SM, length=4)
ax_s1.set_ylabel('Total runoff (Gt d$^{-1}$)', fontsize=SUP_FONT)
ax_s1.legend(handles=band_legend, loc='upper left', bbox_to_anchor=(0.02, 0.82), fontsize=SUP_FONT_SM,
             framealpha=0.92, edgecolor='lightgray')
ax_s1.text(0.02, 0.96, '(a)', transform=ax_s1.transAxes, fontsize=SUP_FONT_PL, fontweight='bold', va='top')
ax_s1.text(0.98, 0.82, '2021 (event)', transform=ax_s1.transAxes, fontsize=SUP_FONT, style='italic',
           ha='right', va='top', color='0.25')

# (b) typical year
ax_s2.set_facecolor('white')
stacked_bars(ax_s2, band_vals_typical, lw=0.15)
ax_s2.tick_params(axis='x', which='minor', length=2, width=0.5, color='gray')
ax_s2.tick_params(axis='both', which='major', labelsize=SUP_FONT_SM, length=4)
ax_s2.set_ylabel('Total runoff (Gt d$^{-1}$)', fontsize=SUP_FONT)
ax_s2.legend(handles=band_legend, loc='upper left', bbox_to_anchor=(0.02, 0.82), fontsize=SUP_FONT_SM,
             framealpha=0.92, edgecolor='lightgray')
ax_s2.text(0.02, 0.96, '(b)', transform=ax_s2.transAxes, fontsize=SUP_FONT_PL, fontweight='bold', va='top')
ax_s2.text(0.98, 0.96, f'{TYPICAL_YEAR} (typical season)', transform=ax_s2.transAxes, fontsize=SUP_FONT,
           style='italic', ha='right', va='top', color='0.25')

# (c) share of JJA runoff in each band
ax_s3.set_facecolor('white')
bar_rows = [('2021', share_2021, 1.6),
            (f'{TYPICAL_YEAR}', share_typical, 0.9),
            ('1950-2020 mean', share_baseline, 0.2)]
for label, shares, y0 in bar_rows:
    left = 0.0
    for i in range(len(ELEV_BANDS)):
        col = ELEV_BANDS[i][2]
        pct = float(shares[i]) * 100.0
        ax_s3.barh(y0, pct, left=left, height=0.36, color=col, edgecolor='black', linewidth=0.55,
                   alpha=0.82, zorder=3)
        if pct >= 6.0:
            if i >= 2:
                text_col = 'white'
            else:
                text_col = 'black'
            ax_s3.text(left + pct / 2.0, y0, f'{pct:.0f}%', ha='center', va='center', fontsize=SUP_FONT_SM,
                       fontweight='bold', color=text_col, zorder=4)
        else:
            ax_s3.text(left + pct + 1.0, y0, f'{pct:.0f}%', ha='left', va='center', fontsize=SUP_FONT_SM,
                       color='black', zorder=4)
        left += pct
    ax_s3.text(-3.5, y0, label, ha='right', va='center', fontsize=SUP_FONT_SM, color='0.25', clip_on=False)

ax_s3.set_xlim(0, 100)
ax_s3.set_ylim(-0.2, 2.05)
ax_s3.set_yticks([])
ax_s3.set_xlabel('Share of JJA total runoff (%)', fontsize=SUP_FONT)
ax_s3.grid(True, axis='x', alpha=0.20, linewidth=0.5)
ax_s3.spines[['top', 'right', 'left']].set_visible(False)
ax_s3.tick_params(axis='x', labelsize=SUP_FONT_SM, length=4)
ax_s3.text(0.02, 0.96, '(c)', transform=ax_s3.transAxes, fontsize=SUP_FONT_PL, fontweight='bold', va='top')

supp_out_path = os.path.join(OUTPUT_DIR, 'extra-fig_august2021_melt.png')
fig_supp.savefig(supp_out_path, dpi=300, bbox_inches='tight', facecolor='white')

client.close()
cluster.close()
