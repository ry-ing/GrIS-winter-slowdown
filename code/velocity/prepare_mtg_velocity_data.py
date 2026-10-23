# W1 anomaly, recovery time and timeseries around M1-M3 (needs SLURM)

import os
import sys
import socket
import glob
import numpy as np
import xarray as xr
import rioxarray
import geopandas as gpd
import pandas as pd
import warnings
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import config

OUTPUT_DIR = os.path.join(config.DATA_DIR, 'velocity_mtg')

MIN_SNR_THRESHOLD = 5.0
BASELINE_START_YEARS = [2016, 2017, 2018, 2019, 2020]

# post-event winters (Nov-Apr)
WINTERS = [
    ('2021-11-01', '2022-04-30'),
    ('2022-11-01', '2023-04-30'),
    ('2023-11-01', '2024-04-30'),
    ('2024-11-01', '2025-04-30'),
]

RECOVERY_THRESHOLD = 0.99

# box half size (m)
BOX_HALF_SIZE = {11: (70_000, 70_000),
                 12: (30_000, 30_000),
                 13: (100_000, 55_000)}

SLURM_CONFIG = {
    'account': 'geos_extra',
    'cores': 8,
    'memory': '32GB',
    'queue': 'General',
    'walltime': '06:00:00',
    'log_directory': '/home/s1834371/scratch/dask_log',
    # workers on the same node as this script, so they can reach the scheduler
    'job_extra_directives': ['--job-name=prep-mtg-data', f'--nodelist={socket.gethostname().split(".")[0]}'],
    'job_script_prologue': ['export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=120s'],
}


# median velocity over a winter
def winter_median(ds_vel, v_var, start, end):
    return ds_vel[v_var].sel(time=slice(start, end)).median(dim='time').compute()


os.makedirs(OUTPUT_DIR, exist_ok=True)

cluster = SLURMCluster(**SLURM_CONFIG)
cluster.scale(jobs=4)
client = Client(cluster)

# marine glaciers, north to south = M1, M2, M3
rois = gpd.read_file(config.ROI_SHAPEFILE_PATH).to_crs('EPSG:3413')
rois['id'] = rois['id'].astype(int)
rois = rois[rois['id'].isin(config.MTG_ROI_LABELS)].copy()
rois['cy'] = rois.geometry.centroid.y
rois = rois.sort_values('cy', ascending=False).reset_index(drop=True)

# velocity, m/yr, SNR > 5
files = sorted(glob.glob(os.path.join(config.VELOCITY_DATA_DIR, '*.nc')))
ds_all = xr.open_mfdataset(files, combine='by_coords', parallel=False, chunks={'time': 1}, engine='netcdf4')
ds_all = ds_all.drop_vars(['time_bnds', 'lat', 'lon', 'lat_bnds', 'lon_bnds'], errors='ignore')
ds_all = ds_all.rio.write_crs('EPSG:3413')
ds_all = ds_all.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)
v_var = 'land_ice_surface_velocity_magnitude'
e_var = 'land_ice_surface_velocity_magnitude_std'

for i in range(len(rois)):
    roi_id = int(rois.iloc[i]['id'])
    name = f'M{i + 1}'
    geom = rois.iloc[i].geometry
    print(name)

    # box around the glacier
    b = geom.bounds
    cx = (b[0] + b[2]) / 2
    cy = (b[1] + b[3]) / 2
    half_w = BOX_HALF_SIZE[roi_id][0]
    half_h = BOX_HALF_SIZE[roi_id][1]
    ds_vel = ds_all.rio.clip_box(minx=cx - half_w, miny=cy - half_h, maxx=cx + half_w, maxy=cy + half_h)
    ds_vel[v_var] = ds_vel[v_var] * 365.25
    ds_vel[e_var] = ds_vel[e_var] * 365.25
    ds_vel = ds_vel.where((ds_vel[v_var] / ds_vel[e_var]) > MIN_SNR_THRESHOLD)

    # baseline and winter medians
    baseline_list = []
    for yr in BASELINE_START_YEARS:
        baseline_list.append(winter_median(ds_vel, v_var, f'{yr}-11-01', f'{yr+1}-04-30'))
    v_baseline = xr.concat(baseline_list, dim='year').median(dim='year')
    post = []
    for start, end in WINTERS:
        post.append(winter_median(ds_vel, v_var, start, end))
    w1 = post[0]

    # W1 anomaly, slowdown only
    anom_w1 = (w1 - v_baseline) / v_baseline * 100.0
    anom_w1 = anom_w1.where(anom_w1 < 0)
    anom_w1.rename('velocity_anomaly_pct').to_netcdf(os.path.join(OUTPUT_DIR, f'spatial_anomaly_w1_{name}.nc'))

    # recovery time in winters (5 = not recovered)
    w1_slowdown_mask = (w1 < v_baseline * RECOVERY_THRESHOLD) & w1.notnull()
    recovery_map = np.full_like(w1, np.nan, dtype=float)
    for k in range(4):
        recovered = post[k] >= v_baseline * RECOVERY_THRESHOLD
        recovery_map = np.where(np.isnan(recovery_map) & recovered, k, recovery_map)
    recovery_map = np.where(np.isnan(recovery_map) & w1_slowdown_mask, 5, recovery_map)
    recovery_map = np.where(w1_slowdown_mask, recovery_map, np.nan)
    recovery_da = xr.DataArray(recovery_map, coords={'x': v_baseline.x, 'y': v_baseline.y}, dims=['y', 'x'],
                               name='recovery_time')
    recovery_da.to_netcdf(os.path.join(OUTPUT_DIR, f'recovery_time_{name}.nc'))

    # timeseries
    sub = ds_vel.sel(x=slice(b[0], b[2]), y=slice(max(b[1], b[3]), min(b[1], b[3])))
    # spatial median over the ROI
    sub = sub.chunk({'x': -1, 'y': -1})
    ts_v = sub[v_var].median(dim=['x', 'y']).compute()
    ts_e = sub[e_var].median(dim=['x', 'y']).compute()
    ts_smooth = ts_v.rolling(time=6, center=True, min_periods=1).mean()
    df = pd.DataFrame()
    df['date'] = ts_v.time.values
    df['velocity'] = ts_v.values
    df['velocity_error'] = ts_e.values
    df['velocity_smooth'] = ts_smooth.values
    out_csv = os.path.join(OUTPUT_DIR, f'ROI_{roi_id}_timeseries.csv')
    df.to_csv(out_csv, index=False)

client.close()
cluster.close()
