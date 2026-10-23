# GrIS-wide slowdown map
# PLOT_ONLY = False to recompute the grids (slow)

import os
import sys
import socket
import gc
import glob
import warnings
import numpy as np
import xarray as xr
import rioxarray
import scipy.ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import cartopy.crs as ccrs

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import config

warnings.filterwarnings('ignore')

PLOT_ONLY = True   # True = skip processing, read cached NetCDFs

CACHE_DIR = config.CACHE_DIR
VEL_DIR = os.path.join(config.DATA_DIR, 'velocity')
OUTPUT_DIR = config.FIG_DIR

# whole GrIS (EPSG:3413)
REGION_BOX = {'minx': -700_000.0, 'miny': -3336959.339030, 'maxx': 900_000.0, 'maxy': -813125.734030}

# baseline winters (Nov-Apr)
BASELINE_WINTERS = [
    ('2016-11-01', '2017-04-30'),
    ('2017-11-01', '2018-04-30'),
    ('2018-11-01', '2019-04-30'),
    ('2019-11-01', '2020-04-30'),
    ('2020-11-01', '2021-04-30'),
]

# post-event winters
POST_WINTERS = [
    ('2021-11-01', '2022-04-30'),
    ('2022-11-01', '2023-04-30'),
    ('2023-11-01', '2024-04-30'),
    ('2024-11-01', '2025-04-30'),
]

MIN_SNR = 5.0           # velocity / uncertainty
MAX_ELEV = 3600.0       # m a.s.l.
MIN_VEL = 50.0          # m/yr
SMOOTH_SIZE = 5         # 5x5 median filter on the anomaly
RECOVERY_THRESH = 0.99  # recovered when back to 99% of baseline

SLURM_CONFIG = {
    'account': 'geos_extra',
    'cores': 8,
    'memory': '30GB',
    'queue': 'General',
    'walltime': '06:00:00',
    'log_directory': '/home/s1834371/scratch/dask_log',
    'job_extra_directives': ['--job-name=gris-wide-fig', f'--nodelist={socket.gethostname().split(".")[0]}'],   # same node as this script
}

PROJ_3413 = ccrs.Stereographic(central_latitude=90, central_longitude=-45, true_scale_latitude=70)
OCEAN_COLOR = '#AFCED9'
LAND_COLOR = "#787878"

# figure size
FIG_W_IN = 25.0 / 2.54
FIG_H_IN = 32.0 / 2.54

FS_CBAR = 8
FS_TICK = 8

# colourbar position
CBAR_RECT = [0.55, 0.1, 0.35, 0.014]


# slice y
def yslice(da, ylo, yhi):
    if float(da.y[0]) > float(da.y[-1]):
        return da.sel(y=slice(yhi, ylo))
    return da.sel(y=slice(ylo, yhi))


# median velocity over a winter
def winter_median(ds, v_var, start, end):
    return ds[v_var].sel(time=slice(start, end)).median(dim='time').compute()


os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not PLOT_ONLY:
    from dask.distributed import Client
    from dask_jobqueue import SLURMCluster

    cluster = SLURMCluster(**SLURM_CONFIG)
    cluster.scale(jobs=4)
    client = Client(cluster)
    client.as_current()

    # velocity, m/yr, SNR > 5
    files = sorted(glob.glob(os.path.join(config.VELOCITY_DATA_DIR, '*.nc')))
    ds = xr.open_mfdataset(files, combine='by_coords', parallel=True, chunks={'time': 1}, engine='netcdf4')
    ds = ds.drop_vars(['time_bnds', 'lat', 'lon', 'lat_bnds', 'lon_bnds'], errors='ignore')
    ds = ds.rio.write_crs('EPSG:3413')
    ds = ds.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)
    ds = ds.rio.clip_box(**REGION_BOX)
    v_var = 'land_ice_surface_velocity_magnitude'
    e_var = 'land_ice_surface_velocity_magnitude_std'
    ds[v_var] = ds[v_var] * 365.25
    ds[e_var] = ds[e_var] * 365.25
    ds = ds.where((ds[v_var] / ds[e_var]) > MIN_SNR)

    bed = xr.open_dataset(config.BEDMACHINE_PATH)
    bed = bed.sel(x=slice(REGION_BOX['minx'] - 10_000, REGION_BOX['maxx'] + 10_000),
                  y=slice(REGION_BOX['maxy'] + 10_000, REGION_BOX['miny'] - 10_000))
    surf_da = bed['surface']

    mask_ds = xr.open_dataset(config.MASK_PATH_FILE).rio.write_crs('EPSG:3413')
    mask_ds = mask_ds.rio.clip_box(**REGION_BOX)
    mask_var = list(mask_ds.data_vars)[0]
    mask_da = xr.where(mask_ds[mask_var] == 2, 1, 0)

    # baseline = median of the baseline winter medians
    bl_list = []
    for s, e in BASELINE_WINTERS:
        bl_list.append(winter_median(ds, v_var, s, e))
    v_base = xr.concat(bl_list, dim='year').median(dim='year')
    v_mean_bl = ds[v_var].sel(time=slice('2015-01-01', '2020-12-31')).mean(dim='time').compute()

    surf_rg = surf_da.interp_like(v_base, method='linear')
    mask_rg = mask_da.interp_like(v_base, method='nearest')
    valid = (mask_rg == 1) & (surf_rg < MAX_ELEV) & (v_mean_bl >= MIN_VEL) & np.isfinite(v_base)

    v_post = []
    for s, e in POST_WINTERS:
        v_post.append(winter_median(ds, v_var, s, e))
    v_w1 = v_post[0]

    # W1 anomaly with the 5x5 median filter
    anom_raw = ((v_w1 - v_base) / v_base * 100.0).where(valid)
    anom_vals = scipy.ndimage.median_filter(np.where(np.isfinite(anom_raw.values), anom_raw.values, 0.0),
                                            size=SMOOTH_SIZE)
    anom_da = xr.DataArray(anom_vals, coords=anom_raw.coords, dims=anom_raw.dims, name='velocity_anomaly_pct')
    anom_da = anom_da.where(valid & np.isfinite(anom_raw))

    # recovery time
    slowed = valid & (anom_da < 0)
    w1_slowdown_mask = slowed & (v_w1 < v_base * RECOVERY_THRESH) & v_w1.notnull()
    recovery_map = np.full_like(v_w1, np.nan, dtype=float)
    for i in range(4):
        recovered = v_post[i] >= v_base * RECOVERY_THRESH
        recovery_map = np.where(np.isnan(recovery_map) & recovered, i, recovery_map)
    recovery_map = np.where(np.isnan(recovery_map) & w1_slowdown_mask, 5, recovery_map)
    recovery_map = np.where(w1_slowdown_mask, recovery_map, np.nan)
    rec = xr.DataArray(recovery_map, coords=v_w1.coords, dims=v_w1.dims, name='recovery_winters')

    anom_da.compute().to_netcdf(os.path.join(VEL_DIR, 'gris_anom_w1.nc'),
                                encoding={'velocity_anomaly_pct': {'zlib': True, 'complevel': 4}})
    rec.compute().to_netcdf(os.path.join(VEL_DIR, 'gris_recovery.nc'),
                            encoding={'recovery_winters': {'zlib': True, 'complevel': 4}})
    surf_da.compute().to_netcdf(os.path.join(CACHE_DIR, 'gris_surf.nc'))
    mask_da.compute().to_netcdf(os.path.join(CACHE_DIR, 'gris_mask.nc'))

    client.close()
    cluster.close()


# figure
surf_da = xr.open_dataarray(os.path.join(CACHE_DIR, 'gris_surf.nc'))[::2, ::2]
mask_da = xr.open_dataarray(os.path.join(CACHE_DIR, 'gris_mask.nc'))[::2, ::2]
anom_da = xr.open_dataarray(os.path.join(VEL_DIR, 'gris_anom_w1.nc'))
rec_da = xr.open_dataarray(os.path.join(VEL_DIR, 'gris_recovery.nc'))

# map extent = anomaly grid + 20 km
xmin = float(anom_da.x.min()) - 20_000
xmax = float(anom_da.x.max()) + 20_000
ymin = float(anom_da.y.min()) - 20_000
ymax = float(anom_da.y.max()) + 20_000

fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), subplot_kw={'projection': PROJ_3413})
ax.set_extent([xmin, xmax, ymin, ymax], crs=PROJ_3413)
ax.set_facecolor(OCEAN_COLOR)

mask_c = yslice(mask_da.sel(x=slice(xmin, xmax)), ymin, ymax)
surf_c = yslice(surf_da.sel(x=slice(xmin, xmax)), ymin, ymax)

# grey land, white see-through ice
land_v = xr.where(surf_c > 1.0, 1.0, np.nan)
ax.pcolormesh(land_v.x, land_v.y, land_v, cmap=mcolors.ListedColormap([LAND_COLOR]), vmin=0, vmax=2,
              transform=PROJ_3413, alpha=1.0, zorder=1)
ice_v = xr.where(mask_c > 0.5, 1.0, np.nan)
ax.pcolormesh(ice_v.x, ice_v.y, ice_v, cmap=mcolors.ListedColormap(['white']), vmin=0, vmax=2,
              transform=PROJ_3413, alpha=0.75, zorder=2)

# coast, margin and contours
ax.contour(surf_c.x, surf_c.y, (surf_c <= 1.0).astype(float), levels=[0.5], colors='#888888',
           linewidths=0.45, transform=PROJ_3413, zorder=5)
ax.contour(mask_c.x, mask_c.y, mask_c, levels=[0.5], colors='black', linewidths=0.08,
           transform=PROJ_3413, zorder=6)
surf_ice = surf_c.where(mask_c.interp_like(surf_c, method='nearest') > 0.5)
ax.contour(surf_ice.x, surf_ice.y, surf_ice, levels=list(range(200, 3600, 200)), colors='black',
           linewidths=0.1, transform=PROJ_3413, zorder=7)

gl = ax.gridlines(draw_labels=True, linewidth=0, x_inline=False, y_inline=False, dms=False)
gl.top_labels = True
gl.bottom_labels = False
gl.left_labels = True
gl.right_labels = False
gl.xlocator = mticker.FixedLocator(range(-90, 20, 20))
gl.ylocator = mticker.FixedLocator(range(60, 90, 5))
gl.xlabel_style = dict(fontsize=FS_TICK - 1, color='black', rotation=0)
gl.ylabel_style = dict(fontsize=FS_TICK - 1, color='black', rotation=90)
del surf_da, surf_c, land_v, ice_v, surf_ice
gc.collect()

# no slowdown / no data
anom_c = yslice(anom_da.sel(x=slice(xmin, xmax)), ymin, ymax)
mask_aligned = mask_c.reindex_like(anom_c, method='nearest')
del anom_da, mask_da, mask_c
gc.collect()

no_slowdown = xr.where((mask_aligned > 0.5) & anom_c.notnull() & (anom_c >= 0), 1.0, np.nan)
ax.pcolormesh(no_slowdown.x, no_slowdown.y, no_slowdown, cmap=mcolors.ListedColormap(['#BDB8B2']),
              vmin=0, vmax=2, transform=PROJ_3413, alpha=0.60, zorder=2.5)
del no_slowdown

no_data = xr.where((mask_aligned > 0.5) & anom_c.isnull(), 1.0, 0.0)
with matplotlib.rc_context({'hatch.color': '#666666', 'hatch.linewidth': 0.4}):
    ax.contourf(no_data.x, no_data.y, no_data, levels=[0.5, 1.5], colors=['none'], hatches=['///////'],
                transform=PROJ_3413, zorder=2.7)
del no_data, mask_aligned
gc.collect()

# the slowdown
anom_plot = anom_c.where(anom_c < 0)
del anom_c
im = ax.pcolormesh(anom_plot.x, anom_plot.y, anom_plot, cmap='YlGnBu', norm=mcolors.Normalize(vmin=-25, vmax=0),
                   transform=PROJ_3413, alpha=0.85, zorder=3)
del anom_plot
gc.collect()

# stippling
rec_multi = yslice(rec_da.where(rec_da > 1).sel(x=slice(xmin, xmax)), ymin, ymax)
d_sub = rec_multi[::15, ::15]
del rec_da, rec_multi
xx, yy = np.meshgrid(d_sub.x.values, d_sub.y.values)
stip = np.isfinite(d_sub.values)
ax.scatter(xx[stip], yy[stip], s=1.5, c='black', marker='.', alpha=0.75, linewidths=0, zorder=4,
           transform=PROJ_3413)
del xx, yy, stip, d_sub
gc.collect()

cax = ax.inset_axes(CBAR_RECT, zorder=11)
cbar = fig.colorbar(im, cax=cax, orientation='horizontal', extend='min')
cbar.set_label('Winter 2021/22 velocity anomaly (%)', fontsize=FS_CBAR, labelpad=2)
cbar.ax.tick_params(labelsize=FS_TICK)

legend_elements = [mpatches.Patch(facecolor='#BDB8B2', alpha=0.75, edgecolor='#888888', linewidth=0.5,
                                  label='No slowdown'),
                   mpatches.Patch(facecolor='none', edgecolor='#444444', hatch='///////', linewidth=0.8,
                                  label='No data')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=FS_TICK - 1, framealpha=0.88,
          edgecolor='#888888', handlelength=1.8, handleheight=1.2)

out = os.path.join(OUTPUT_DIR, 'FigS6_GrIS_slowdown_map.png')
plt.savefig(out, dpi=800, bbox_inches='tight')
plt.close(fig)
