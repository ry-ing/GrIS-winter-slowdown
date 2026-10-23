# W1-W4 anomaly grids, elevation/thickness bins and ROI recovery
# run prepare_land-term_velocity_data.py and pick the winter windows first

import os
import sys
import socket
import glob
import json
import warnings
import numpy as np
import xarray as xr
import rioxarray
import geopandas as gpd
import pandas as pd
import scipy.ndimage
from scipy import stats
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import config

warnings.filterwarnings("ignore")

VELOCITY_DATA_DIR = config.VELOCITY_DATA_DIR
MASK_PATH_FILE = config.MASK_PATH_FILE
BEDMACHINE_PATH = config.BEDMACHINE_PATH
ROI_SHAPEFILE_PATH = config.ROI_SHAPEFILE_PATH
OUTPUT_DIR = os.path.join(config.DATA_DIR, 'scatter_plots')

# winter windows for the ROIs
MANUAL_WINDOWS_PATH = os.path.join(config.DATA_DIR, 'velocity', 'winter_windows.json')

# ROI timeseries
VELOCITY_TS_DIR = os.path.join(config.DATA_DIR, 'velocity')

MIN_SNR_THRESHOLD = 5.0
MAX_ELEVATION = 1600.0
MIN_ANNUAL_VEL = 50.0
SMOOTH_SIZE = 5
BASELINE_START_YEARS = [2016, 2017, 2018, 2019, 2020]

WINTERS = {
    'W1': ('2021-11-01', '2022-04-30'),
    'W2': ('2022-11-01', '2023-04-30'),
    'W3': ('2023-11-01', '2024-04-30'),
    'W4': ('2024-11-01', '2025-04-30'),
}

SELECTED_ROI_LABELS = config.ROI_LABELS

# load 10 km extra for the median filter
xmin, xmax, ymin, ymax = config.STUDY_EXTENT
LOAD_PAD_M = 10_000.0
REGION_BOX = {'minx': xmin - LOAD_PAD_M, 'miny': ymin - LOAD_PAD_M,
              'maxx': xmax + LOAD_PAD_M, 'maxy': ymax + LOAD_PAD_M}

ELEV_BIN_WIDTH = 50
THICK_BIN_WIDTH = 100
MIN_BIN_PIXELS = 20

SLURM_CONFIG = {
    'account': 'geos_extra',
    'cores': 8,
    'memory': '30GB',
    'queue': 'General',
    'walltime': '06:00:00',
    'log_directory': '/home/s1834371/scratch/dask_log',
    'job_extra_directives': ['--job-name=prep-plot-data', f'--nodelist={socket.gethostname().split(".")[0]}'],
    'scheduler_options': {'port': 8786, 'dashboard_address': ':8787'},
    'job_script_prologue': ['export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=120s'],
}


# crop to the study extent (y can run either way)
def crop_to_study_extent(da):
    da = da.sel(x=slice(xmin, xmax))
    if float(da.y.values[0]) > float(da.y.values[-1]):
        return da.sel(y=slice(ymax, ymin))
    return da.sel(y=slice(ymin, ymax))


# median velocity over a winter
def winter_median(ds_vel, v_var, start, end):
    return ds_vel[v_var].sel(time=slice(start, end)).median(dim='time').compute()


# 5x5 median filter
def apply_median_filter(raw_anomaly_da, valid_mask):
    filtered_vals = scipy.ndimage.median_filter(raw_anomaly_da.fillna(0).values, size=SMOOTH_SIZE)
    filtered = xr.DataArray(filtered_vals, coords=raw_anomaly_da.coords, dims=raw_anomaly_da.dims)
    return filtered.where(valid_mask & ~np.isnan(raw_anomaly_da.values))


# binned anomalies
def compute_bins(predictor, edges, anom_v):
    rows = []
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        in_bin = (predictor >= lo) & (predictor < hi)
        w1_valid = anom_v['W1'][in_bin]
        w1_valid = w1_valid[np.isfinite(w1_valid)]
        if len(w1_valid) < MIN_BIN_PIXELS:
            continue
        row = {'bin_centre': float(0.5 * (lo + hi))}
        for label in ['W1', 'W2', 'W3', 'W4']:
            vals = anom_v[label][in_bin]
            vals = vals[np.isfinite(vals)]
            if len(vals) >= 3:
                row[f'{label}_mean'] = float(np.mean(vals))
                row[f'{label}_std'] = float(np.std(vals))
                row[f'{label}_n'] = int(len(vals))
            else:
                row[f'{label}_mean'] = np.nan
                row[f'{label}_std'] = np.nan
                row[f'{label}_n'] = 0
        rows.append(row)
    return pd.DataFrame(rows)


# median velocity_smooth between two dates
def window_median_smooth(ts_df, start, end):
    sub = ts_df.loc[(ts_df['date'] >= pd.Timestamp(start)) & (ts_df['date'] <= pd.Timestamp(end)), 'velocity_smooth']
    if sub.empty:
        return np.nan
    return float(sub.median())


os.makedirs(OUTPUT_DIR, exist_ok=True)

cluster = SLURMCluster(**SLURM_CONFIG)
cluster.scale(jobs=4)
client = Client(cluster)

# ROI shapefile
rois = gpd.read_file(ROI_SHAPEFILE_PATH).to_crs('EPSG:3413')
rois['orig_label'] = rois['id'].astype(int)
rois = rois[rois['orig_label'].isin(SELECTED_ROI_LABELS)].copy()

# BedMachine
bed_ds = xr.open_dataset(BEDMACHINE_PATH)
bed_ds = bed_ds.sel(x=slice(REGION_BOX['minx'] - 10_000, REGION_BOX['maxx'] + 10_000),
                    y=slice(REGION_BOX['maxy'] + 10_000, REGION_BOX['miny'] - 10_000))
surf_da = bed_ds['surface']
thick_da = bed_ds['thickness']

# ice mask
ice_mask_ds = xr.open_dataset(MASK_PATH_FILE).rio.write_crs('EPSG:3413')
ice_mask_ds = ice_mask_ds.rio.clip_box(**REGION_BOX)
mask_var = list(ice_mask_ds.data_vars)[0]
mask_binary = xr.where(ice_mask_ds[mask_var] == 2, 1, 0)

# velocity, m/yr, SNR > 5
print('loading velocity')
files = sorted(glob.glob(os.path.join(VELOCITY_DATA_DIR, '*.nc')))
ds_vel = xr.open_mfdataset(files, combine='by_coords', parallel=False, chunks={'time': 1}, engine='netcdf4')
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
baseline_list = []
for yr in BASELINE_START_YEARS:
    baseline_list.append(winter_median(ds_vel, v_var, f'{yr}-11-01', f'{yr+1}-04-30'))
baseline_da = xr.concat(baseline_list, dim='year').median(dim='year')

mean_annual_v = ds_vel[v_var].sel(time=slice('2015-01-01', '2019-12-31')).mean(dim='time').compute()

# BedMachine onto the velocity grid
surf_regrid = surf_da.interp_like(baseline_da, method='linear')
thick_regrid = thick_da.interp_like(baseline_da, method='linear')
mask_regrid = mask_binary.interp_like(baseline_da, method='nearest')

valid_mask = (surf_regrid < MAX_ELEVATION) & (mean_annual_v > MIN_ANNUAL_VEL) & (mask_regrid == 1)

# post-event winter medians
w = {}
for label in WINTERS:
    start = WINTERS[label][0]
    end = WINTERS[label][1]
    w[label] = winter_median(ds_vel, v_var, start, end)

# filtered anomaly for each winter
anom = {}
for label in WINTERS:
    raw = ((w[label] - baseline_da) / baseline_da * 100.0).where(valid_mask)
    anom[label] = apply_median_filter(raw, valid_mask)

# crop to the study extent (after filtering)
for label in WINTERS:
    anom[label] = crop_to_study_extent(anom[label])
valid_mask = crop_to_study_extent(valid_mask)
surf_regrid = crop_to_study_extent(surf_regrid)
thick_regrid = crop_to_study_extent(thick_regrid)

# slowed = valid and W1 < 0
slowed_mask = valid_mask & (anom['W1'] < 0)
print('slowed pixels', int(slowed_mask.values.sum()), 'of', int(valid_mask.values.sum()))

# save grids
for label in WINTERS:
    out = os.path.join(OUTPUT_DIR, f'anomaly_{label}.nc')
    anom[label].where(valid_mask).rename(f'anomaly_{label}').to_netcdf(out)

# unfiltered winter medians, for the baseline std
meds = []
for b in baseline_list:
    meds.append(crop_to_study_extent(b))
meds.append(crop_to_study_extent(w['W1']))
winter_meds = xr.concat(meds, dim='winter')
winter_names = [f'{yr}/{yr+1-2000:02d}' for yr in BASELINE_START_YEARS] + ['W1']
winter_meds = winter_meds.assign_coords(winter=winter_names)
winter_meds = winter_meds.where(valid_mask).astype('float32').rename('winter_median_velocity')
winter_meds.to_netcdf(os.path.join(OUTPUT_DIR, 'winter_median_velocity.nc'),
                      encoding={'winter_median_velocity': {'zlib': True, 'complevel': 4}})

surf_regrid.where(valid_mask).rename('surface_elevation').to_netcdf(os.path.join(OUTPUT_DIR, 'surface_elevation.nc'))
thick_regrid.where(valid_mask).rename('ice_thickness').to_netcdf(os.path.join(OUTPUT_DIR, 'ice_thickness.nc'))
slowed_mask.rename('slowed_mask').to_netcdf(os.path.join(OUTPUT_DIR, 'valid_slowed_mask.nc'))

# slowed pixels for binning
slowed_flat = slowed_mask.values.ravel().astype(bool)
elev_flat = surf_regrid.values.ravel()
thick_flat = thick_regrid.values.ravel()
use = slowed_flat & np.isfinite(elev_flat) & np.isfinite(thick_flat)

elev_v = elev_flat[use]
thick_v = thick_flat[use]
anom_v = {}
for label in WINTERS:
    anom_v[label] = anom[label].values.ravel()[use]

# bins
elev_edges = np.arange(0, MAX_ELEVATION + ELEV_BIN_WIDTH, ELEV_BIN_WIDTH)
elev_df = compute_bins(elev_v, elev_edges, anom_v)
elev_df.to_csv(os.path.join(OUTPUT_DIR, 'elevation_bins.csv'), index=False)

thick_max = float(np.nanpercentile(thick_v, 99))
thick_edges = np.arange(0, thick_max + THICK_BIN_WIDTH, THICK_BIN_WIDTH)
thick_df = compute_bins(thick_v, thick_edges, anom_v)
thick_df.to_csv(os.path.join(OUTPUT_DIR, 'thickness_bins.csv'), index=False)


# ROI recovery rates, using the manual winter windows
with open(MANUAL_WINDOWS_PATH) as f:
    manual_windows = json.load(f)

roi_rows = []
for i, roi_row in rois.iterrows():
    lbl = int(roi_row['orig_label'])

    windows = manual_windows[str(lbl)]
    baseline_windows = windows[:5]   # 2016-2020
    event_windows = windows[5:]      # W1-W4

    ts_df = pd.read_csv(os.path.join(VELOCITY_TS_DIR, f'ROI_{lbl}_timeseries.csv'), parse_dates=['date'])

    bl_vals = []
    for s, e in baseline_windows:
        v = window_median_smooth(ts_df, s, e)
        if np.isfinite(v):
            bl_vals.append(v)
    baseline_vel = float(np.median(bl_vals))

    w_vels = {}
    w_anoms = {}
    for k in range(4):
        label = ['W1', 'W2', 'W3', 'W4'][k]
        w_vels[label] = window_median_smooth(ts_df, event_windows[k][0], event_windows[k][1])
        w_anoms[label] = (w_vels[label] - baseline_vel) / baseline_vel * 100.0

    # recovery slope
    winter_indices = np.array([1., 2., 3., 4.])
    winter_vels = np.array([w_vels['W1'], w_vels['W2'], w_vels['W3'], w_vels['W4']])
    n_fit = 4
    for k in range(4):
        if np.isfinite(winter_vels[k]) and winter_vels[k] >= baseline_vel * 0.99:
            n_fit = k + 1
            break
    indices_to_fit = winter_indices[:n_fit]
    vels_to_fit = winter_vels[:n_fit]

    ok = np.isfinite(vels_to_fit)
    if ok.sum() >= 2:
        fit = stats.linregress(indices_to_fit[ok], vels_to_fit[ok])
        slope = fit.slope
        r2 = fit.rvalue ** 2
        p_val = fit.pvalue
        se = fit.stderr
    else:
        slope = np.nan
        r2 = np.nan
        p_val = np.nan
        se = np.nan

    # normalised slope
    initial_anom = baseline_vel - w_vels['W1']
    if initial_anom > 0:
        norm_slope = slope / initial_anom
    else:
        norm_slope = np.nan

    roi_rows.append({
        'roi_label': lbl,
        'baseline_vel': baseline_vel,
        'W1_vel': w_vels['W1'],
        'W2_vel': w_vels['W2'],
        'W3_vel': w_vels['W3'],
        'W4_vel': w_vels['W4'],
        'W1_anom_pct': w_anoms['W1'],
        'W2_anom_pct': w_anoms['W2'],
        'W3_anom_pct': w_anoms['W3'],
        'W4_anom_pct': w_anoms['W4'],
        'recovery_slope': slope,
        'recovery_slope_norm': norm_slope,
        'recovery_r2': r2,
        'recovery_p': p_val,
        'recovery_slope_se': se,
    })
    print('L' + str(lbl), round(w_anoms['W1'], 1), '% slope', round(slope, 2))

roi_df = pd.DataFrame(roi_rows).sort_values('roi_label')
roi_df.to_csv(os.path.join(OUTPUT_DIR, 'roi_recovery.csv'), index=False)
print('done')

client.close()
cluster.close()
