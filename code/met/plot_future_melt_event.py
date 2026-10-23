# Largest future melt event in MAR-CESM2 SSP3-7.0 (2060-2099)

import os
import sys
import socket
import numpy as np
import xarray as xr
import rioxarray
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
from matplotlib.colors import LightSource
from scipy.stats import gaussian_kde
from scipy.ndimage import distance_transform_edt
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import warnings

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import config

warnings.filterwarnings("ignore")

MAR_FILE = os.path.join(config.MAR_CESM2_DIR, 'MARv3.14.3-10km-daily-ERA5-{year}.nc')
OUTPUT_DIR = config.FIG_DIR
CACHE_DIR = config.CACHE_DIR

# MAR region
MIN_X = -340000.0
MIN_Y = -2750000.0
MAX_X = 50000.0
MAX_Y = -2000000.0
MAR_REGION = {'minx': MIN_X, 'miny': MIN_Y, 'maxx': MAX_X, 'maxy': MAX_Y}
PROJ_3413 = ccrs.Stereographic(central_latitude=90, central_longitude=-45, true_scale_latitude=70)

FUTURE_YEARS = range(2060, 2100)
PIXEL_AREA_M2 = (10e3) ** 2
N_DAYS_JJA = 92
OCEAN_COLOR = '#AED9E0'

MAP_XMIN = -340000
MAP_XMAX = -50000
ROI_EXTENT_PAD_M = 40_000.0
PANEL_A_VMAX_CM_DAY = 11.0

# elevation bands: min, max, colour, label
ELEV_BANDS = [
    (0,    500,  '#C6DBEF', '0–500 m'),
    (500,  1000, '#6BAED6', '500–1000 m'),
    (1000, 1500, '#2171B5', '1000–1500 m'),
    (1500, 2000, '#08306B', '1500-2000 m'),
]

# caches
REFRESH = False
PANEL_B_CACHE = os.path.join(CACHE_DIR, 'future_panel_b_elevation_bands_jja.nc')
PANEL_C_CACHE = os.path.join(CACHE_DIR, 'future_panel_c_scatter_2060_2099_jjas.npz')

SLURM_CONFIG = {
    "account": "geos_extra",
    "cores": 8,
    "memory": "32GB",
    "queue": "General",
    "walltime": "04:00:00",
    "log_directory": '/home/s1834371/scratch/dask_log',
    "job_extra_directives": ["--job-name=dask-melt-fig", "--output=/dev/null", "--error=/dev/null",
                             f"--nodelist={socket.gethostname().split('.')[0]}"],   # same node as this script
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


# MAR MSK or SH
def open_static(year, var):
    ds = xr.open_dataset(MAR_FILE.format(year=year))[[var]]
    ds = ds.assign_coords(x=np.round(ds.x.values * 1000).astype(np.float64),
                          y=np.round(ds.y.values * 1000).astype(np.float64))
    ds = ds.rio.set_spatial_dims(x_dim='x', y_dim='y')
    ds = ds.rio.write_crs("EPSG:3413")
    ds = ds.rio.clip_box(**MAR_REGION)
    return ds[var]


# map extent from the ROIs
rois = gpd.read_file(config.ROI_SHAPEFILE_PATH).to_crs('EPSG:3413')
rois = rois[rois['id'].astype(int).isin(config.ROI_LABELS)]
bounds = rois.total_bounds
MAP_YMIN = float(bounds[1] - ROI_EXTENT_PAD_M)
MAP_YMAX = float(bounds[3] + ROI_EXTENT_PAD_M)

# reference grids
bed_ds = xr.open_dataset(config.BEDMACHINE_PATH)
bed_ds = bed_ds.sel(x=slice(MAP_XMIN - 10000, MAP_XMAX + 10000), y=slice(MAP_YMAX + 10000, MAP_YMIN - 10000))
surface_bedm = bed_ds['surface']

mask_ds = xr.open_dataset(config.MASK_PATH_FILE).rio.write_crs("EPSG:3413")
mask_ds = mask_ds.rio.clip_box(minx=MIN_X - 20000, miny=MIN_Y - 20000, maxx=MAX_X + 20000, maxy=MAX_Y + 20000)
mask_hires = xr.where(mask_ds['Band1'] == 2, 1, 0)
mask_on_bedm = mask_hires.interp_like(surface_bedm, method='nearest').fillna(0)

# MAR mask and surface
ref = open_ru(2060)
mask_on_mar = (open_static(2060, 'MSK') > 50).astype(float).interp_like(ref.isel(time=0), method='nearest').fillna(0)
surface_on_mar = open_static(2021, 'SH')


# panel (c) first, the event is picked from it
if os.path.exists(PANEL_C_CACHE) and not REFRESH:
    c = np.load(PANEL_C_CACHE)
    scatter_runoff = c['scatter_runoff']
    scatter_delta = c['scatter_delta']
    scatter_dates = c['scatter_dates_ns'].astype('datetime64[ns]')
else:
    all_runoff = []
    all_delta = []
    all_dates = []
    for yr in FUTURE_YEARS:
        ru = open_ru(yr).where(mask_on_mar == 1).sel(time=slice(f'{yr}-06-01', f'{yr}-09-30'))
        ts = (ru.sum(dim=['x', 'y']) * PIXEL_AREA_M2 * 1e-3 * 1000 / 1e12).compute()
        vals = ts.values.astype(float)
        times = pd.DatetimeIndex(ts.time.values).normalize()
        # 3-day increase
        for i in range(3, len(vals)):
            all_runoff.append(vals[i])
            all_delta.append(vals[i] - np.mean(vals[i - 3:i]))
            all_dates.append(times[i])
        print(yr)
    all_delta = np.array(all_delta)
    pos = all_delta > 0
    scatter_runoff = np.array(all_runoff)[pos]
    scatter_delta = all_delta[pos]
    scatter_dates = np.array(all_dates).astype('datetime64[ns]')[pos]
    np.savez(PANEL_C_CACHE, scatter_runoff=scatter_runoff, scatter_delta=scatter_delta,
             scatter_dates_ns=scatter_dates.astype('int64'))

# the event = largest 3-day increase
i_event = int(np.argmax(scatter_delta))
EVENT = pd.Timestamp(scatter_dates[i_event])
EVENT_YEAR = EVENT.year
event_x = scatter_runoff[i_event]
event_y = scatter_delta[i_event]
print('largest increase', round(event_y, 2), 'Gt/d on', EVENT.date())


# (a) runoff change
ru_event = open_ru(EVENT_YEAR).where(mask_on_mar == 1)
pre3 = ru_event.sel(time=slice((EVENT - pd.Timedelta(days=3)).strftime('%Y-%m-%d'),
                               (EVENT - pd.Timedelta(days=1)).strftime('%Y-%m-%d'))).mean(dim='time').compute()
on_day = ru_event.sel(time=EVENT.strftime('%Y-%m-%d')).squeeze().compute()
delta = (on_day - pre3).where(mask_on_mar == 1)

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
runoff_change_plot = xr.DataArray(vals, coords=delta_on_bed.coords, dims=delta_on_bed.dims).where(ice)


# (b) elevation bands
if os.path.exists(PANEL_B_CACHE) and not REFRESH:
    with xr.open_dataset(PANEL_B_CACHE) as c:
        band_vals = c['band_vals'].values.astype(float)
else:
    ru_jja = ru_event.sel(time=slice(f'{EVENT_YEAR}-06-01', f'{EVENT_YEAR}-08-31'))
    band_vals = []
    for band in ELEV_BANDS:
        ru_band = ru_jja.where((surface_on_mar >= band[0]) & (surface_on_mar < band[1]))
        total = ru_band.sum(dim=['x', 'y']) * PIXEL_AREA_M2 * 1e-3 * 1000 / 1e12
        band_vals.append(total.compute().values)
    band_vals = np.array(band_vals).astype(float)
    ds = xr.Dataset({'band_vals': (('band', 'day'), band_vals)})
    ds = ds.assign_coords(elev_min=('band', np.array([b[0] for b in ELEV_BANDS], dtype=np.int32)))
    ds = ds.assign_coords(elev_max=('band', np.array([b[1] for b in ELEV_BANDS], dtype=np.int32)))
    ds.to_netcdf(PANEL_B_CACHE)
event_day_x = (EVENT - pd.Timestamp(f'{EVENT_YEAR}-06-01')).days


# figure
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

# (a) map
ax_map.set_extent([MAP_XMIN, MAP_XMAX, MAP_YMIN, MAP_YMAX], crs=PROJ_3413)
ax_map.set_facecolor(OCEAN_COLOR)

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
im = ax_map.pcolormesh(change_cm.x, change_cm.y, change_cm, cmap='YlGnBu_r',
                       norm=mcolors.Normalize(vmin=0.0, vmax=PANEL_A_VMAX_CM_DAY, clip=True),
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

# (b) elevation bands in the event year
ax_elev.set_facecolor('white')
bottom = np.zeros(N_DAYS_JJA)
for i in range(len(ELEV_BANDS)):
    ax_elev.bar(np.arange(N_DAYS_JJA), band_vals[i], bottom=bottom, color=ELEV_BANDS[i][2], edgecolor='black',
                linewidth=0.2, width=1.0, align='center', label=ELEV_BANDS[i][3], zorder=2)
    bottom += band_vals[i]
ax_elev.axvline(event_day_x, color='black', linewidth=2.0, linestyle=':', zorder=5)
ax_elev.set_xlim(-0.5, N_DAYS_JJA - 0.5)
ax_elev.set_ylim(bottom=0)
ax_elev.text(event_day_x + 1.0, ax_elev.get_ylim()[1] * 0.97, EVENT.strftime('%d %B %Y'), fontsize=FONT_SM,
             va='top', ha='left', style='italic')
ax_elev.set_xticks([0, 30, 61])   # 1 Jun, 1 Jul, 1 Aug
ax_elev.set_xticklabels(['Jun', 'Jul', 'Aug'])
ax_elev.set_xticks(np.arange(0, N_DAYS_JJA), minor=True)
ax_elev.tick_params(axis='x', which='minor', length=3, width=0.6, color='gray')
ax_elev.tick_params(axis='x', which='major', length=6)
ax_elev.set_ylabel('Total Runoff (Gt d$^{-1}$)')
ax_elev.grid(True, axis='y', alpha=0.25, linewidth=0.5)
ax_elev.spines[['top', 'right']].set_visible(False)
band_legend = []
for band in ELEV_BANDS:
    band_legend.append(mpatches.Patch(facecolor=band[2], edgecolor='black', linewidth=0.5, label=band[3]))
ax_elev.legend(handles=band_legend, loc='upper left', bbox_to_anchor=(0.02, 0.90),
               framealpha=0.92, edgecolor='lightgray')
ax_elev.text(0.02, 0.97, '(b)', transform=ax_elev.transAxes, fontsize=FONT_PL, fontweight='bold', va='top')

# (c) scatter
ax_roc.set_facecolor('white')
xy = np.vstack([scatter_runoff, scatter_delta])
density = gaussian_kde(xy)(xy)
density_norm = (density - density.min()) / (density.max() - density.min())
sc = ax_roc.scatter(scatter_runoff, scatter_delta, s=80, alpha=0.7, linewidths=0.4, edgecolors='#444444',
                    zorder=2, c=density_norm, cmap='Greens')
ax_roc.scatter(event_x, event_y, s=360, facecolors='none', edgecolors='black', linewidths=2.3, zorder=6)
ax_roc.axhline(0, color='gray', linewidth=0.8, linestyle='--', zorder=1)
ax_roc.set_xlabel('Total Daily Runoff (Gt d$^{-1}$)')
ax_roc.set_ylabel('Change in runoff (Gt)')
ax_roc.grid(True, alpha=0.2, linewidth=0.5)
ax_roc.spines[['top', 'right']].set_visible(False)
ax_roc.text(0.02, 0.97, '(c)', transform=ax_roc.transAxes, fontsize=FONT_PL, fontweight='bold', va='top')

cbar_c = fig.colorbar(sc, cax=ax_roc.inset_axes([0.03, 0.58, 0.032, 0.30]), orientation='vertical')
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
ticks = np.linspace(0.0, PANEL_A_VMAX_CM_DAY, 6)
cbar.set_ticks(ticks)
cbar.set_ticklabels([f'{v:.2g}' for v in ticks])
cbar.set_label('Increase in runoff (cm day$^{-2}$)', fontsize=FONT, labelpad=4)
cbar.ax.tick_params(labelsize=FONT_SM - 2, length=4)

out_path = os.path.join(OUTPUT_DIR, 'FigS11_future_melt-event.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)

client.close()
cluster.close()
