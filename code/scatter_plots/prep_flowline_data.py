# Flowlines up from each ROI seed point, with the anomaly along them
# run prepare_vel_scatter_data.py first

import os
import sys
import socket
import glob
import warnings
import numpy as np
import xarray as xr
import rioxarray
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import config

warnings.filterwarnings("ignore")

SEED_SHAPEFILE_PATH = os.path.join(config.DATA_DIR, 'roi_shapefiles', 'roi_seed_points.shp')
VELOCITY_DATA_DIR = config.VELOCITY_DATA_DIR
BEDMACHINE_PATH = config.BEDMACHINE_PATH
DATA_DIR = os.path.join(config.DATA_DIR, 'scatter_plots')
OUTPUT_DIR = DATA_DIR
PLOT_DIR = os.path.join(config.FIG_DIR, 'flowlines')

SELECTED_ROI_LABELS = config.ROI_LABELS
WINTERS = ['W1', 'W2', 'W3', 'W4']

# year for flow direction
VELOCITY_YEAR = '2021'

STEP_M = 100           # step length along the flowline (m)
MAX_DIST_M = 80_000    # longest flowline allowed (m)

# stop here
MAX_ELEV_M = 1500.0

# stop if stagnant (after 5 km)
ABS_MIN_SPEED_MYR = 5.0
MIN_DIST_BEFORE_SPEED_CHECK_M = 5_000

# smoothing of vx/vy (pixels)
SMOOTH_SIGMA = 3

MIN_SNR = 5.0

# study extent + 10 km
xmin, xmax, ymin, ymax = config.STUDY_EXTENT
REGION_BOX = {'minx': xmin - 10_000, 'miny': ymin - 10_000,
              'maxx': xmax + 10_000, 'maxy': ymax + 10_000}

SLURM_CONFIG = {
    'account': 'geos_extra',
    'cores': 8,
    'memory': '30GB',
    'queue': 'General',
    'walltime': '06:00:00',
    'log_directory': '/home/s1834371/scratch/dask_log',
    'job_extra_directives': ['--job-name=prep-plot-data', f'--nodelist={socket.gethostname().split(".")[0]}'],
    'scheduler_options': {'port': 8786, 'dashboard_address': ':8787'},
}


# bilinear interpolator
def make_interp(values, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    v = np.asarray(values, dtype=float)
    if y[0] > y[-1]:
        order = np.argsort(y)
        y = y[order]
        v = v[order, :]
    if x[0] > x[-1]:
        order = np.argsort(x)
        x = x[order]
        v = v[:, order]
    v = np.where(np.isfinite(v), v, 0.0)
    return RegularGridInterpolator((y, x), v, method='linear', bounds_error=False, fill_value=np.nan)


# value of an interpolator at points x, y (arrays)
def interp_at(rgi, x, y):
    pts = np.column_stack([np.atleast_1d(y), np.atleast_1d(x)])
    return rgi(pts)


# gaussian smoothing ignoring NaNs
def smooth_component(da, sigma):
    vals = da.values.astype(float)
    weight = np.where(np.isfinite(vals), 1.0, 0.0)
    filled = np.where(np.isfinite(vals), vals, 0.0)
    sm_val = gaussian_filter(filled, sigma=sigma)
    sm_wt = gaussian_filter(weight, sigma=sigma)
    result = np.where(sm_wt > 0.01, sm_val / sm_wt, np.nan)
    return xr.DataArray(result, coords=da.coords, dims=da.dims)


# upstream direction and speed at x, y
def unit_upstream(x, y):
    vx = float(interp_at(vx_interp, x, y)[0])
    vy = float(interp_at(vy_interp, x, y)[0])
    spd = np.hypot(vx, vy)
    if spd < 1e-3 or not np.isfinite(spd):
        return 0.0, 0.0, 0.0
    return -vx / spd, -vy / spd, spd


# one RK4 step upstream
def rk4_step_upstream(cx, cy):
    dx1, dy1, spd = unit_upstream(cx, cy)
    if spd < 1e-3:
        return np.nan, np.nan, spd
    dx2, dy2, s2 = unit_upstream(cx + 0.5 * STEP_M * dx1, cy + 0.5 * STEP_M * dy1)
    dx3, dy3, s3 = unit_upstream(cx + 0.5 * STEP_M * dx2, cy + 0.5 * STEP_M * dy2)
    dx4, dy4, s4 = unit_upstream(cx + STEP_M * dx3, cy + STEP_M * dy3)
    new_x = cx + STEP_M / 6.0 * (dx1 + 2*dx2 + 2*dx3 + dx4)
    new_y = cy + STEP_M / 6.0 * (dy1 + 2*dy2 + 2*dy3 + dy4)
    return new_x, new_y, spd


# mean velocity across a 500 m corridor
def sample_velocity_corridor(x, y):
    vx_c = float(interp_at(vx_interp, x, y)[0])
    vy_c = float(interp_at(vy_interp, x, y)[0])
    speed_c = np.hypot(vx_c, vy_c)
    if not np.isfinite(speed_c) or speed_c < 1e-3:
        return np.nan, np.nan, np.nan, np.nan

    # perpendicular to the flow direction
    perp_x = -vy_c / speed_c
    perp_y = vx_c / speed_c

    vx_samples = []
    vy_samples = []
    v_samples = []
    for offset in np.arange(-250, 250 + 100, 100):
        vx_s = float(interp_at(vx_interp, x + offset * perp_x, y + offset * perp_y)[0])
        vy_s = float(interp_at(vy_interp, x + offset * perp_x, y + offset * perp_y)[0])
        if np.isfinite(vx_s) and np.isfinite(vy_s):
            vx_samples.append(vx_s)
            vy_samples.append(vy_s)
            v_samples.append(np.hypot(vx_s, vy_s))

    if len(vx_samples) == 0:
        return np.nan, np.nan, np.nan, np.nan
    return np.mean(vx_samples), np.mean(vy_samples), np.mean(v_samples), np.std(v_samples)


# trace flowline upstream
def trace_flowline_upstream(seed_x, seed_y):
    vx0 = float(interp_at(vx_interp, seed_x, seed_y)[0])
    vy0 = float(interp_at(vy_interp, seed_x, seed_y)[0])
    xs = [seed_x]
    ys = [seed_y]
    dists = [0.0]
    speeds = [float(np.hypot(vx0, vy0))]
    cx = seed_x
    cy = seed_y

    while dists[-1] < MAX_DIST_M:
        nx, ny, spd = rk4_step_upstream(cx, cy)

        if not np.isfinite(nx) or not np.isfinite(ny):
            break

        # reached max elevation
        elev_new = float(interp_at(surf_interp, nx, ny)[0])
        if np.isfinite(elev_new) and elev_new >= MAX_ELEV_M:
            cx = nx
            cy = ny
            dists.append(dists[-1] + STEP_M)
            xs.append(cx)
            ys.append(cy)
            speeds.append(spd)
            break

        on_ice = float(interp_at(mask_interp, nx, ny)[0])
        if not np.isfinite(on_ice) or on_ice < 0.5:
            break

        current_dist = dists[-1] + STEP_M
        if current_dist > MIN_DIST_BEFORE_SPEED_CHECK_M and spd < ABS_MIN_SPEED_MYR:
            break

        cx = nx
        cy = ny
        dists.append(current_dist)
        xs.append(cx)
        ys.append(cy)
        speeds.append(spd)

    return np.array(xs), np.array(ys), np.array(dists), np.array(speeds)


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

cluster = SLURMCluster(**SLURM_CONFIG)
cluster.scale(jobs=4)
client = Client(cluster)

# seed points
seeds = gpd.read_file(SEED_SHAPEFILE_PATH).to_crs('EPSG:3413')
seeds['roi_label'] = seeds['id'].astype(int)
seeds = seeds[seeds['roi_label'].isin(SELECTED_ROI_LABELS)].copy()

# 2021 mean velocity, m/yr, SNR > 5
print('loading velocity')
files = sorted(glob.glob(os.path.join(VELOCITY_DATA_DIR, '*.nc')))
ds = xr.open_mfdataset(files, combine='by_coords', parallel=False, chunks={'time': 1}, engine='netcdf4')
ds = ds.drop_vars(['time_bnds', 'lat', 'lon', 'lat_bnds', 'lon_bnds'], errors='ignore')
ds = ds.rio.write_crs('EPSG:3413')
ds = ds.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)
ds = ds.rio.clip_box(**REGION_BOX)

v_var = 'land_ice_surface_velocity_magnitude'
e_var = 'land_ice_surface_velocity_magnitude_std'
vx_var = 'land_ice_surface_easting_velocity'
vy_var = 'land_ice_surface_northing_velocity'
ds[v_var] = ds[v_var] * 365.25
ds[e_var] = ds[e_var] * 365.25
ds[vx_var] = ds[vx_var] * 365.25
ds[vy_var] = ds[vy_var] * 365.25

snr_mask = (ds[v_var] / ds[e_var]) > MIN_SNR
ds[vx_var] = ds[vx_var].where(snr_mask)
ds[vy_var] = ds[vy_var].where(snr_mask)
ds[v_var] = ds[v_var].where(snr_mask)

year_slice = slice(f'{VELOCITY_YEAR}-01-01', f'{VELOCITY_YEAR}-12-31')
vx_da = ds[vx_var].sel(time=year_slice).mean(dim='time').compute()
vy_da = ds[vy_var].sel(time=year_slice).mean(dim='time').compute()
v_da = ds[v_var].sel(time=year_slice).mean(dim='time').compute()

# smooth vx, vy (Hvidberg et al. 2020)
vx_sm = smooth_component(vx_da, SMOOTH_SIGMA)
vy_sm = smooth_component(vy_da, SMOOTH_SIGMA)

x_coords = vx_da.x.values
y_coords = vx_da.y.values
vx_interp = make_interp(vx_sm.values, x_coords, y_coords)
vy_interp = make_interp(vy_sm.values, x_coords, y_coords)
v_interp = make_interp(v_da.values, x_coords, y_coords)

# BedMachine
bed_ds = xr.open_dataset(BEDMACHINE_PATH)
bed_ds = bed_ds.sel(x=slice(REGION_BOX['minx'] - 20_000, REGION_BOX['maxx'] + 20_000),
                    y=slice(REGION_BOX['maxy'] + 20_000, REGION_BOX['miny'] - 20_000))
bm_x = bed_ds.x.values
bm_y = bed_ds.y.values
surf_interp = make_interp(bed_ds['surface'].values, bm_x, bm_y)
thick_interp = make_interp(bed_ds['thickness'].values, bm_x, bm_y)
mask_interp = make_interp((bed_ds['mask'].values == 2).astype(float), bm_x, bm_y)

# anomaly grids from prepare_vel_scatter_data.py
anom_interps = {}
for label in WINTERS:
    da = xr.open_dataarray(os.path.join(DATA_DIR, f'anomaly_{label}.nc'))
    anom_interps[label] = make_interp(da.values, da.x.values, da.y.values)


# trace the flowlines
seeds = seeds.sort_values('roi_label')
for i, seed_row in seeds.iterrows():
    lbl = int(seed_row['roi_label'])
    seed_x = float(seed_row.geometry.x)
    seed_y = float(seed_row.geometry.y)

    xs, ys, dists, speeds = trace_flowline_upstream(seed_x, seed_y)
    n_pts = len(xs)
    print('L' + str(lbl), n_pts, 'points,', dists[-1] / 1000, 'km')

    # values along the flowline
    elev_pts = interp_at(surf_interp, xs, ys)
    thick_pts = interp_at(thick_interp, xs, ys)

    vx_pts = []
    vy_pts = []
    v_pts = []
    v_std_pts = []
    for k in range(n_pts):
        vx, vy, v, v_std = sample_velocity_corridor(xs[k], ys[k])
        vx_pts.append(vx)
        vy_pts.append(vy)
        v_pts.append(v)
        v_std_pts.append(v_std)

    df = pd.DataFrame()
    df['distance_m'] = dists
    df['x_3413'] = xs
    df['y_3413'] = ys
    df['surface_elev'] = elev_pts
    df['ice_thickness'] = thick_pts
    df['vx_mean'] = vx_pts
    df['vy_mean'] = vy_pts
    df['v_mean'] = v_pts
    df['v_std'] = v_std_pts
    for label in WINTERS:
        df[f'{label}_anom'] = interp_at(anom_interps[label], xs, ys)
    for label in WINTERS:
        df[f'{label}_slowed'] = df[f'{label}_anom'] < 0

    df.to_csv(os.path.join(OUTPUT_DIR, f'flowline_ROI_{lbl}.csv'), index=False)

    # plot flowline
    buffer_m = 30_000
    x_mask = (x_coords >= xs.min() - buffer_m) & (x_coords <= xs.max() + buffer_m)
    y_mask = (y_coords >= ys.min() - buffer_m) & (y_coords <= ys.max() + buffer_m)
    xx, yy = np.meshgrid(x_coords[x_mask], y_coords[y_mask])
    v_cropped = v_da.values[np.ix_(y_mask, x_mask)]
    vmin, vmax = np.nanpercentile(v_cropped[np.isfinite(v_cropped)], [5, 95])

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.pcolormesh(xx, yy, v_cropped, shading='auto', cmap='viridis',
                       norm=Normalize(vmin=vmin, vmax=vmax), alpha=0.85)
    plt.colorbar(im, ax=ax, label='Annual Ice Velocity (m/yr)')

    # flowline coloured by distance from the seed
    distances = np.sqrt((xs - xs[0])**2 + (ys - ys[0])**2)
    line = ax.scatter(xs, ys, c=distances / 1000, s=40, cmap='plasma',
                      edgecolors='black', linewidth=0.7, zorder=5, label='Flowline')
    plt.colorbar(line, ax=ax, label='Distance from seed (km)', pad=0.15)

    ax.plot(seed_x, seed_y, 'r*', markersize=30, markeredgecolor='white',
            markeredgewidth=2.0, zorder=6, label='Seed point')
    ax.set_xlabel('X coordinate (EPSG:3413, m)', fontsize=12)
    ax.set_ylabel('Y coordinate (EPSG:3413, m)', fontsize=12)
    ax.set_title(f'ROI {lbl}: Calculated Flowline with Annual Ice Velocity (2021)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')

    plot_path = os.path.join(PLOT_DIR, f'flowline_velocity_ROI_{lbl}.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

print('done')

client.close()
cluster.close()
