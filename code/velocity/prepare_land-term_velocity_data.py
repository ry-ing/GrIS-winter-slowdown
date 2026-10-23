# W1 anomaly, recovery time and ROI timeseries for the land-terminating ROIs (needs SLURM)

import os
import sys
import glob
import socket
import dask
import numpy as np
import xarray as xr
import rioxarray
import geopandas as gpd
import pandas as pd
import scipy.ndimage
import warnings
from pyproj import Transformer
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import config

OUTPUT_DIR = os.path.join(config.DATA_DIR, 'velocity')

MIN_SNR_THRESHOLD = 5.0
MAX_ELEVATION = 1600.0
MIN_ANNUAL_VEL = 50.0
SMOOTH_SIZE = 5            # 5x5 median filter on the anomaly map
BASELINE_START_YEARS = [2016, 2017, 2018, 2019, 2020]

# post-event winters (Nov-Apr)
WINTERS = [
    ('2021-11-01', '2022-04-30'),
    ('2022-11-01', '2023-04-30'),
    ('2023-11-01', '2024-04-30'),
    ('2024-11-01', '2025-04-30'),
]

# recovery threshold
RECOVERY_THRESHOLD = 0.99

# whole GrIS
REGION_BOX = {'minx': -700000.0, 'miny': -3400000.0, 'maxx': 900000.0, 'maxy': -600000.0}

SLURM_CONFIG = {
    'account': 'geos_extra',
    'cores': 8,
    'processes': 2,          # 2 workers per job, ~10 GB each
    'memory': '20GB',        # 3 jobs x 20 GB + 32 GB main job + interactive session < 128 GB limit
    'queue': 'General',
    'walltime': '06:00:00',
    'log_directory': '/home/s1834371/scratch/dask_log',
    # workers on the same node as this script, so they can reach the scheduler
    'job_extra_directives': ['--job-name=prep-plot-data', f'--nodelist={socket.gethostname().split(".")[0]}'],
    'scheduler_options': {'port': 8786, 'dashboard_address': ':8787'},
    'job_script_prologue': ['export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=120s'],
}


# median velocity over a winter (lazy), whole winter in one chunk
def winter_median(ds_vel, v_var, start, end):
    da = ds_vel[v_var].sel(time=slice(start, end)).chunk({'time': -1, 'x': 750, 'y': 1350})
    return da.median(dim='time')


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(config.CACHE_DIR, exist_ok=True)

dask.config.set({'distributed.comm.timeouts.connect': '120s'})
# p2p rechunk hangs if a worker gets restarted, tasks method doesn't
dask.config.set({'array.rechunk.method': 'tasks'})
cluster = SLURMCluster(**SLURM_CONFIG)
cluster.scale(jobs=3)
client = Client(cluster)
client.wait_for_workers(1)


# keep the netCDF file cache small, workers run out of memory otherwise
def small_file_cache():
    import netCDF4
    import xarray
    netCDF4.set_chunk_cache(8 * 1024**2)
    xarray.set_options(file_cache_maxsize=16)


client.register_worker_callbacks(setup=small_file_cache)

rois = gpd.read_file(config.ROI_SHAPEFILE_PATH).to_crs('EPSG:3413')
rois['orig_label'] = rois['id'].astype(int)

# BedMachine
bed_ds = xr.open_dataset(config.BEDMACHINE_PATH)
bed_ds = bed_ds.sel(x=slice(REGION_BOX['minx'] - 10_000, REGION_BOX['maxx'] + 10_000),
                    y=slice(REGION_BOX['maxy'] + 10_000, REGION_BOX['miny'] - 10_000))
bed_ds['surface'].to_netcdf(os.path.join(config.CACHE_DIR, 'bed_surface.nc'))

ice_mask_ds = xr.open_dataset(config.MASK_PATH_FILE).rio.write_crs('EPSG:3413')
ice_mask_ds = ice_mask_ds.rio.clip_box(**REGION_BOX)
mask_var = list(ice_mask_ds.data_vars)[0]
mask_binary = xr.where(ice_mask_ds[mask_var] == 2, 1, 0)
mask_binary.to_netcdf(os.path.join(config.CACHE_DIR, 'ice_mask.nc'))

# velocity, m/yr, SNR > 5
files = sorted(glob.glob(os.path.join(config.VELOCITY_DATA_DIR, '*.nc')))
# tiles matching the file chunks (750 x 1350) to save memory
ds_vel = xr.open_mfdataset(files, combine='by_coords', parallel=False, chunks={'time': 1, 'x': 1500, 'y': 2700},
                           engine='netcdf4')
ds_vel = ds_vel.drop_vars(['time_bnds', 'lat', 'lon', 'lat_bnds', 'lon_bnds'], errors='ignore')
ds_vel = ds_vel.rio.write_crs('EPSG:3413')
ds_vel = ds_vel.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)
ds_vel = ds_vel.rio.clip_box(**REGION_BOX)
v_var = 'land_ice_surface_velocity_magnitude'
e_var = 'land_ice_surface_velocity_magnitude_std'
ds_vel[v_var] = ds_vel[v_var] * 365.25
ds_vel[e_var] = ds_vel[e_var] * 365.25
ds_vel = ds_vel.where((ds_vel[v_var] / ds_vel[e_var]) > MIN_SNR_THRESHOLD)

# baseline and winter medians
to_compute = []
for yr in BASELINE_START_YEARS:
    to_compute.append(winter_median(ds_vel, v_var, f'{yr}-11-01', f'{yr+1}-04-30'))
to_compute.append(ds_vel[v_var].sel(time=slice('2015-01-01', '2019-12-31')).mean(dim='time'))
for start, end in WINTERS:
    to_compute.append(winter_median(ds_vel, v_var, start, end))
results = dask.compute(*to_compute)

baseline_list = list(results[0:5])
mean_annual_v = results[5]
w1 = results[6]
w2 = results[7]
w3 = results[8]
w4 = results[9]
baseline_da = xr.concat(baseline_list, dim='year').median(dim='year')

surf_regrid = bed_ds['surface'].interp_like(baseline_da, method='linear')
mask_regrid = mask_binary.interp_like(baseline_da, method='nearest')
valid_mask = (surf_regrid < MAX_ELEVATION) & (mean_annual_v > MIN_ANNUAL_VEL) & (mask_regrid == 1)

# W1 anomaly with the 5x5 median filter
anomaly_raw = ((w1 - baseline_da) / baseline_da * 100.0).where(valid_mask)
anom_vals = scipy.ndimage.median_filter(anomaly_raw.fillna(0).values, size=SMOOTH_SIZE)
anom_final = xr.DataArray(anom_vals, coords=anomaly_raw.coords, dims=anomaly_raw.dims, name='velocity_anomaly_pct')
anom_final = anom_final.where(valid_mask & ~np.isnan(anomaly_raw.values))

anom_final.to_netcdf(os.path.join(OUTPUT_DIR, 'spatial_anomaly_w1.nc'),
                     encoding={'velocity_anomaly_pct': {'zlib': True, 'complevel': 4}})
valid_mask = valid_mask.rename('valid_mask')
valid_mask.to_netcdf(os.path.join(OUTPUT_DIR, 'valid_mask.nc'), encoding={'valid_mask': {'zlib': True, 'complevel': 4}})

# recovery time (5 = not recovered)
slowed_mask = valid_mask & (anom_final < 0)
w1_slowdown_mask = slowed_mask & (w1 < baseline_da * RECOVERY_THRESHOLD) & w1.notnull()

recovery_map = np.full_like(w1, np.nan, dtype=float)
post_winters = [w1, w2, w3, w4]
for i in range(4):
    recovered = post_winters[i] >= baseline_da * RECOVERY_THRESHOLD
    # winters spent below baseline before this one
    recovery_map = np.where(np.isnan(recovery_map) & recovered, i, recovery_map)
recovery_map = np.where(np.isnan(recovery_map) & w1_slowdown_mask, 5, recovery_map)
recovery_map = np.where(w1_slowdown_mask, recovery_map, np.nan)

recovery = xr.DataArray(recovery_map, coords=w1.coords, dims=w1.dims, name='recovery_winters')
recovery.to_netcdf(os.path.join(OUTPUT_DIR, 'recovery_time.nc'), encoding={'recovery_winters': {'zlib': True, 'complevel': 4}})

# ROI metadata
to_latlon = Transformer.from_crs('EPSG:3413', 'EPSG:4326', always_xy=True)
meta_rows = []
for i, row in rois.iterrows():
    cx = row.geometry.centroid.x
    cy = row.geometry.centroid.y
    lon, lat = to_latlon.transform(cx, cy)
    meta_rows.append({'label': row['orig_label'], 'cx': cx, 'cy': cy, 'lat': lat, 'lon': lon})
pd.DataFrame(meta_rows).to_csv(os.path.join(OUTPUT_DIR, 'roi_metadata.csv'), index=False)

# ROI timeseries
selected_rois = rois[rois['orig_label'].isin(config.ROI_LABELS)].sort_values('orig_label')

# all ROIs in one compute so the files are only read once
labels = []
to_compute = []
for i, row in selected_rois.iterrows():
    bounds = row.geometry.bounds
    y_north = max(bounds[1], bounds[3])
    y_south = min(bounds[1], bounds[3])
    subset = ds_vel.sel(x=slice(bounds[0], bounds[2]), y=slice(y_north, y_south))
    labels.append(row['orig_label'])
    # spatial median over the ROI
    subset = subset.chunk({'x': -1, 'y': -1})
    to_compute.append(subset[v_var].median(dim=['x', 'y']))
    to_compute.append(subset[e_var].median(dim=['x', 'y']))
results = dask.compute(*to_compute)

for k in range(len(labels)):
    lbl = labels[k]
    ts_v = results[2 * k]
    ts_e = results[2 * k + 1]
    ts_smooth = ts_v.rolling(time=6, center=True, min_periods=1).mean()

    df = pd.DataFrame()
    df['date'] = ts_v.time.values
    df['velocity'] = ts_v.values
    df['velocity_error'] = ts_e.values
    df['velocity_smooth'] = ts_smooth.values
    out_csv = os.path.join(OUTPUT_DIR, f'ROI_{lbl}_timeseries.csv')
    df.to_csv(out_csv, index=False)


client.close()
cluster.close()
