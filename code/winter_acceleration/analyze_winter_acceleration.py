# Within-winter acceleration (not used in the paper)

import os
import sys
import glob
import socket
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import geopandas as gpd
import scipy.ndimage
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import config

# valid mask
EXISTING_VALID_MASK_PATH = os.path.join(config.DATA_DIR, 'velocity', 'valid_mask.nc')
OUT_DIR = os.path.join(config.DATA_DIR, 'winter_acceleration')

SELECTED_ROI_LABELS = config.ROI_LABELS

# domain: ROI cluster + buffer
NS_BUFFER_M = 20_000.0
WEST_BUFFER_M = 40_000.0
INLAND_REACH_M = 300_000.0

MIN_SNR_THRESHOLD = 5.0
MAX_ELEVATION = 1600.0

SMOOTH_SPATIAL_SIZE = 9      # pixels
SMOOTH_TEMPORAL_WINDOW = 3   # time steps (~33-45 days)

# winters (Nov start year)
WINTER_START_YEARS = list(range(2016, 2025))

# search windows
AUTUMN_SEARCH = ('{Y}-08-15', '{Y1}-01-15')     # start of winter
SPRING_SEARCH = ('{Y1}-01-01', '{Y1}-07-15')    # end of winter
FULL_WINDOW = ('{Y}-08-15', '{Y1}-07-15')       # both, used for the line fit

# spring speed-up threshold (m/yr per day)
SLOPE_THRESHOLD_M_PER_YR_PER_DAY = 0.5

# workers on the same node
hostname = socket.gethostname().split('.')[0]
SLURM_CONFIG = {
    'account': 'geos_extra',
    'cores': 8,
    'memory': '30GB',
    'queue': 'General',
    'walltime': '04:00:00',
    'log_directory': '/home/s1834371/scratch/dask_log',
    'job_extra_directives': ['--job-name=winter-accel', f'--nodelist={hostname}'],
    'scheduler_options': {'port': 8786, 'dashboard_address': ':8787'},
}


# 2D median filter on one velocity field
def spatial_median(field):
    return scipy.ndimage.median_filter(field, size=SMOOTH_SPATIAL_SIZE, mode='nearest')


# start/end of winter and acceleration
def detect_one_winter(smoothed, Y):
    Y1 = Y + 1
    spatial_shape = smoothed.shape[1:]

    # start = autumn minimum
    autumn = smoothed.sel(time=slice(AUTUMN_SEARCH[0].format(Y=Y), AUTUMN_SEARCH[1].format(Y1=Y1)))
    if autumn.sizes['time'] == 0:
        nan = np.full(spatial_shape, np.nan)
        nat = np.full(spatial_shape, np.datetime64('NaT'))
        return dict(start_date=nat, end_date=nat, v_start=nan, v_end=nan,
                    pct_acceleration=nan, linear_slope=nan, r_squared=nan)
    start_date = autumn.idxmin(dim='time', skipna=True)
    v_start = autumn.min(dim='time', skipna=True).values

    # end = spring speed-up
    spring = smoothed.sel(time=slice(SPRING_SEARCH[0].format(Y1=Y1), SPRING_SEARCH[1].format(Y1=Y1)))
    vals = spring.values.astype(float)
    times = spring['time'].values
    if len(times) < 3:
        end_date = np.full(spatial_shape, np.datetime64('NaT'))
        v_end = np.full(spatial_shape, np.nan)
    else:
        dt_days = np.diff(times).astype('timedelta64[s]').astype(float) / 86400.0
        dt_days = dt_days.reshape((len(dt_days),) + (1,) * len(spatial_shape))
        slope = np.diff(vals, axis=0) / dt_days

        fast = (slope[:-1] > SLOPE_THRESHOLD_M_PER_YR_PER_DAY) & (slope[1:] > SLOPE_THRESHOLD_M_PER_YR_PER_DAY)
        finite = np.isfinite(slope[:-1]) & np.isfinite(slope[1:])
        step_times = times[:-2]
        after_start = step_times.reshape((len(step_times),) + (1,) * len(spatial_shape)) > \
            np.broadcast_to(start_date.values, spatial_shape)[None, ...]
        combined = fast & finite & after_start

        found = combined.any(axis=0)
        first_idx = np.argmax(combined, axis=0)
        idx_grid = np.indices(spatial_shape)
        end_date = np.where(found, step_times[first_idx], np.datetime64('NaT'))
        v_end = np.where(found, vals[:-2][(first_idx,) + tuple(idx_grid)], np.nan)

    pct_acceleration = (v_end - v_start) / v_start * 100.0

    # line fit
    full = smoothed.sel(time=slice(FULL_WINDOW[0].format(Y=Y), FULL_WINDOW[1].format(Y1=Y1)))
    fvals = full.values.astype(float)
    time_shape = (full.sizes['time'],) + (1,) * len(spatial_shape)
    fdays = full['time'].values.astype('datetime64[D]').astype(float).reshape(time_shape)
    x = np.broadcast_to(fdays, fvals.shape)

    t64 = full['time'].values.reshape(time_shape).astype('datetime64[D]')
    start_b = np.broadcast_to(start_date.values, fvals.shape[1:])[None, ...].astype('datetime64[D]')
    end_b = np.broadcast_to(end_date, fvals.shape[1:])[None, ...].astype('datetime64[D]')
    in_window = (t64 >= start_b) & (t64 <= end_b) & np.isfinite(fvals)

    xm = np.where(in_window, x, np.nan)
    ym = np.where(in_window, fvals, np.nan)
    n = np.sum(in_window, axis=0).astype(float)
    sx = np.nansum(xm, axis=0)
    sy = np.nansum(ym, axis=0)
    sxx = np.nansum(xm * xm, axis=0)
    sxy = np.nansum(xm * ym, axis=0)

    denom = n * sxx - sx * sx
    ok = (n >= 2) & (denom != 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        slope_fit = np.where(ok, (n * sxy - sx * sy) / denom, np.nan)
        intercept = np.where(ok, (sy - slope_fit * sx) / n, np.nan)

    y_pred = slope_fit[None, ...] * xm + intercept[None, ...]
    ss_res = np.nansum((ym - y_pred) ** 2, axis=0)
    y_mean = sy / np.where(n > 0, n, np.nan)
    ss_tot = np.nansum((ym - y_mean[None, ...]) ** 2, axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        r_squared = np.where((n >= 2) & (ss_tot != 0), 1.0 - ss_res / ss_tot, np.nan)

    return dict(start_date=start_date.values, end_date=end_date, v_start=v_start, v_end=v_end,
                pct_acceleration=pct_acceleration, linear_slope=slope_fit, r_squared=r_squared)


os.makedirs(OUT_DIR, exist_ok=True)

cluster = SLURMCluster(**SLURM_CONFIG)
cluster.scale(jobs=2)
client = Client(cluster)

# domain from the ROIs
rois = gpd.read_file(config.ROI_SHAPEFILE_PATH).to_crs('EPSG:3413')
rois['orig_label'] = rois['id'].astype(int)
minx, miny, maxx, maxy = rois[rois['orig_label'].isin(SELECTED_ROI_LABELS)].total_bounds
region_box = {'minx': minx - WEST_BUFFER_M, 'miny': miny - NS_BUFFER_M,
              'maxx': minx + INLAND_REACH_M, 'maxy': maxy + NS_BUFFER_M}

# velocity, m/yr, SNR > 5
files = sorted(glob.glob(os.path.join(config.VELOCITY_DATA_DIR, '*.nc')))
ds_vel = xr.open_mfdataset(files, combine='by_coords', parallel=False, chunks={'time': 1}, engine='netcdf4')
ds_vel = ds_vel.drop_vars(['time_bnds', 'lat', 'lon', 'lat_bnds', 'lon_bnds'], errors='ignore')
ds_vel = ds_vel.rio.write_crs('EPSG:3413')
ds_vel = ds_vel.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)
ds_vel = ds_vel.rio.clip_box(**region_box)
v_var = 'land_ice_surface_velocity_magnitude'
e_var = 'land_ice_surface_velocity_magnitude_std'
ds_vel[v_var] = ds_vel[v_var] * 365.25
ds_vel[e_var] = ds_vel[e_var] * 365.25
ds_vel = ds_vel.where((ds_vel[v_var] / ds_vel[e_var]) > MIN_SNR_THRESHOLD)

# valid pixels
template = ds_vel[v_var].isel(time=0)
if os.path.exists(EXISTING_VALID_MASK_PATH):
    valid_mask = xr.open_dataarray(EXISTING_VALID_MASK_PATH).astype(bool)
    valid_mask = valid_mask.interp_like(template, method='nearest').fillna(False)
else:
    bed_ds = xr.open_dataset(config.BEDMACHINE_PATH)
    bed_ds = bed_ds.sel(x=slice(region_box['minx'] - 10_000, region_box['maxx'] + 10_000),
                        y=slice(region_box['maxy'] + 10_000, region_box['miny'] - 10_000))
    surf_regrid = bed_ds['surface'].interp_like(template, method='linear')
    ice_mask_ds = xr.open_dataset(config.MASK_PATH_FILE).rio.write_crs('EPSG:3413')
    ice_mask_ds = ice_mask_ds.rio.clip_box(**region_box)
    mask_var = list(ice_mask_ds.data_vars)[0]
    mask_regrid = xr.where(ice_mask_ds[mask_var] == 2, 1, 0).interp_like(template, method='nearest')
    valid_mask = (surf_regrid < MAX_ELEVATION) & (mask_regrid == 1)
valid_mask = valid_mask.compute()

# smoothing
v = ds_vel[v_var].chunk({'y': -1, 'x': -1})
v_smooth = xr.apply_ufunc(spatial_median, v, input_core_dims=[['y', 'x']], output_core_dims=[['y', 'x']],
                          vectorize=True, dask='parallelized', output_dtypes=[v.dtype])
v_smooth = v_smooth.compute()
v_smooth = v_smooth.rolling(time=SMOOTH_TEMPORAL_WINDOW, center=True, min_periods=1).mean()
v_smooth = v_smooth.where(valid_mask)

per_year_results = []
for Y in WINTER_START_YEARS:
    print(Y)
    per_year_results.append(detect_one_winter(v_smooth, Y))

# save the pixel results as (year, y, x)
out_ds = xr.Dataset(coords={'year': WINTER_START_YEARS, 'y': v_smooth.y, 'x': v_smooth.x})
for key in ['v_start', 'v_end', 'pct_acceleration', 'linear_slope', 'r_squared', 'start_date', 'end_date']:
    out_ds[key] = (('year', 'y', 'x'), np.stack([r[key] for r in per_year_results]))
out_path = os.path.join(OUT_DIR, 'winter_timing_and_acceleration.nc')
out_ds.to_netcdf(out_path)

# domain mean for each winter
domain_rows = []
for i in range(len(WINTER_START_YEARS)):
    Y = WINTER_START_YEARS[i]
    pct = per_year_results[i]['pct_acceleration']
    domain_rows.append({'winter_start_year': Y,
                        'winter_label': f'{Y}/{str(Y+1)[2:]}',
                        'mean_pct_acceleration': float(np.nanmean(pct)),
                        'median_pct_acceleration': float(np.nanmedian(pct)),
                        'n_pixels': int(np.isfinite(pct).sum())})
domain_csv = os.path.join(OUT_DIR, 'domain_mean_winter_acceleration_by_year.csv')
pd.DataFrame(domain_rows).to_csv(domain_csv, index=False)

# ROI mean curves
sel_rois = rois[rois['orig_label'].isin(SELECTED_ROI_LABELS)].sort_values('orig_label')
roi_curves = {}
for i, row in sel_rois.iterrows():
    b = row.geometry.bounds
    y0 = min(b[1], b[3])
    y1 = max(b[1], b[3])
    curve = v_smooth.sel(x=slice(b[0], b[2]), y=slice(y1, y0)).mean(dim=['x', 'y'], skipna=True)
    roi_curves[int(row['orig_label'])] = curve

roi_rows = []
for Y in WINTER_START_YEARS:
    for lbl in SELECTED_ROI_LABELS:
        curve = roi_curves[lbl]
        res = detect_one_winter(curve, Y)
        # 0-d arrays
        start_date = pd.Timestamp(res['start_date'][()])
        end_date = pd.Timestamp(res['end_date'][()])

        # curvature
        curvature = np.nan
        if pd.notna(start_date) and pd.notna(end_date):
            seg = curve.sel(time=slice(start_date, end_date))
            t_days = seg['time'].values.astype('datetime64[D]').astype(float)
            vals = seg.values.astype(float)
            ok = np.isfinite(t_days) & np.isfinite(vals)
            if ok.sum() >= 4:
                curvature = float(np.polyfit(t_days[ok], vals[ok], deg=2)[0])

        roi_rows.append({'roi_label': lbl,
                         'winter_start_year': Y,
                         'winter_label': f'{Y}/{str(Y+1)[2:]}',
                         'start_date': start_date,
                         'end_date': end_date,
                         'v_start': float(res['v_start']),
                         'v_end': float(res['v_end']),
                         'pct_acceleration': float(res['pct_acceleration']),
                         'linear_slope_m_yr_per_day': float(res['linear_slope']),
                         'quadratic_curvature': curvature,
                         'r_squared': float(res['r_squared'])})

roi_csv = os.path.join(OUT_DIR, 'roi_winter_acceleration_summary.csv')
pd.DataFrame(roi_rows).to_csv(roi_csv, index=False)

# save ROI curves
roi_curve_rows = []
for lbl in roi_curves:
    c = roi_curves[lbl].compute()
    for t, val in zip(c['time'].values, c.values):
        roi_curve_rows.append({'roi_label': lbl, 'time': t, 'velocity_smooth': float(val)})
pd.DataFrame(roi_curve_rows).to_csv(os.path.join(OUT_DIR, 'roi_smoothed_curves.csv'), index=False)

client.close()
cluster.close()
