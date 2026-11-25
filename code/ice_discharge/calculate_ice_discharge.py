# Ice discharge through the flux gates (drawn manually in QGIS, flux_gates_manual.gpkg)

import os
import sys
import socket
import glob
import math
import time
import warnings
import gc
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rioxarray
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource
from shapely.geometry import LineString
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import config

VELOCITY_DATA_DIR = config.VELOCITY_DATA_DIR
BEDMACHINE_PATH = config.BEDMACHINE_PATH
GIMP_DEM_PATH = config.GIMP_DEM_PATH
MASK_PATH_FILE = config.MASK_PATH_FILE
THINNING_PATH = config.THINNING_PATH
ROI_SHAPEFILE_PATH = config.ROI_SHAPEFILE_PATH
OUTPUT_DIR = os.path.join(config.DATA_DIR, 'ice_discharge')
FIGURE_DIR = config.FIG_DIR
MANUAL_GATES_PATH = os.path.join(OUTPUT_DIR, 'flux_gates_manual.gpkg')

SELECTED_ROI_LABELS = config.ROI_LABELS

RHO_ICE = 917.0   # kg m-3 ice densituy
PIXEL_W = 200.0   # m for s1 promice velocities v4
REGION_BOX = {'minx': -320_000.0, 'miny': -2_620_000.0, 'maxx': -130_000.0, 'maxy': -2_270_000.0}
MIN_SNR_THRESHOLD = 5.0
MIN_GATE_COVERAGE = 0.70   # skip timestep if less of the gate has velocity

# True = hold thickness at Bedmachine so discharge only changes with velocity
# False = GIMP 2007 thickness + Khan et al. thinning so that ice thickness changes
FIXED_THICKNESS = True

EVENT_DATE = pd.Timestamp('2021-08-14')
PLOT_XMIN = pd.Timestamp('2016-01-01')
PLOT_XMAX = pd.Timestamp('2025-03-01')

SLURM_CONFIG = {
    'account': 'geos_extra',
    'cores': 8,
    'memory': '30GB',
    'queue': 'General',
    'walltime': '03:00:00',
    'log_directory': '/home/s1834371/scratch/dask_log',
    'job_extra_directives': ['--job-name=ice-discharge', f'--nodelist={socket.gethostname().split(".")[0]}'],
}

# datastore reads sometimes fail
def compute_with_retry(lazy, name):
    for attempt in range(1, 6):
        try:
            return lazy.compute()
        except (RuntimeError, OSError) as e:
            if attempt == 5:
                raise
            print('read failed, trying again', name)
            time.sleep(15 * min(attempt, 4))


# clip to box
def clip(da, xmin, xmax, ymin, ymax):
    da = da.sel(x=slice(xmin, xmax))
    return da.isel(y=(da.y.values >= ymin) & (da.y.values <= ymax))


# azimuth from p1 to p2, clockwise from north
def azimuth(p1, p2):
    return (-math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) + 90) % 360


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs('/home/s1834371/scratch/dask_log', exist_ok=True)

cluster = SLURMCluster(**SLURM_CONFIG)
cluster.scale(jobs=4)
client = Client(cluster)


# velocity, m/yr, SNR > 5
print('loading velocity')
vx_var = 'land_ice_surface_easting_velocity'
vy_var = 'land_ice_surface_northing_velocity'
v_var = 'land_ice_surface_velocity_magnitude'
e_var = 'land_ice_surface_easting_velocity_std'

# parallel=False, datastore unreliable otherwise
files = sorted(glob.glob(os.path.join(VELOCITY_DATA_DIR, '*.nc')))
ds_vel = xr.open_mfdataset(files, combine='by_coords', parallel=False, chunks={'time': 1}, engine='netcdf4')
ds_vel = ds_vel.drop_vars(['time_bnds'])
ds_vel = ds_vel.rio.write_crs('EPSG:3413')
ds_vel = ds_vel.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)
ds_vel = ds_vel.rio.clip_box(**REGION_BOX)
for var in [vx_var, vy_var, v_var, e_var]:
    ds_vel[var] = ds_vel[var] * 365.25
snr = ds_vel[v_var] / ds_vel[e_var].where(ds_vel[e_var] > 0)
ds_vel = ds_vel.where(snr > MIN_SNR_THRESHOLD)


# other data
bm = xr.open_dataset(BEDMACHINE_PATH)
bed = bm['bed'].squeeze()
errbed = bm['errbed'].squeeze()
H_bm = bm['thickness'].squeeze()

# 2007 thickness = GIMP surface - BedMachine bed
gimp = rioxarray.open_rasterio(GIMP_DEM_PATH).squeeze()
gimp.rio.write_crs('EPSG:3413', inplace=True)
gimp.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)
H_2007 = gimp - bed.interp(x=gimp.x, y=gimp.y, method='nearest')
H_2007 = H_2007.where(H_2007 > 0, 0.0)

mask = xr.where(xr.open_dataset(MASK_PATH_FILE)['Band1'] == 2, 1, 0).squeeze()

rois = gpd.read_file(ROI_SHAPEFILE_PATH).to_crs('EPSG:3413')
rois['label'] = rois['id'].astype(int)
rois = rois[rois['label'].isin(SELECTED_ROI_LABELS)].sort_values('label').reset_index(drop=True)


# gates drawn in QGIS
gates = gpd.read_file(MANUAL_GATES_PATH).to_crs('EPSG:3413')
x_arr = ds_vel.x.values
y_arr = ds_vel.y.values

gate_pixels = {}
gate_az = {}
gate_lines = {}
for i, row in gates.iterrows():
    lbl = int(row['roi_label'])
    line = row.geometry

    # gate pixels
    n_pts = max(int(line.length / PIXEL_W) + 2, 2)
    pixels = []
    for k in range(n_pts):
        pt = line.interpolate(k / (n_pts - 1) * line.length)
        px = float(x_arr[np.argmin(np.abs(x_arr - pt.x))])
        py = float(y_arr[np.argmin(np.abs(y_arr - pt.y))])
        if (px, py) not in pixels:
            pixels.append((px, py))

    # gate 5 drawn the wrong way round
    if lbl == 5:
        az5 = azimuth(line.coords[0], line.coords[-1])
        if az5 < 150 or az5 > 200:
            line = LineString(list(line.coords)[::-1])

    # azimuth for each pixel
    c = list(line.coords)
    seg_az = []
    seg_mid = []
    for k in range(len(c) - 1):
        seg_az.append(azimuth(c[k], c[k + 1]))
        seg_mid.append(((c[k][0] + c[k + 1][0]) / 2, (c[k][1] + c[k + 1][1]) / 2))
    az = []
    for gx, gy in pixels:
        dists = [math.sqrt((gx - mx)**2 + (gy - my)**2) for mx, my in seg_mid]
        az.append(seg_az[int(np.argmin(dists))])

    gate_pixels[lbl] = pixels
    gate_az[lbl] = az
    gate_lines[lbl] = LineString(pixels)
    print('gate', lbl, len(pixels), 'pixels')


# map of the gates over mean speed
pad = 40_000.0
xmin = rois.total_bounds[0] - pad
xmax = rois.total_bounds[2] + pad + 45_000.0
ymin = rois.total_bounds[1] - pad
ymax = rois.total_bounds[3] + pad

speed = compute_with_retry(clip(ds_vel[v_var], xmin, xmax, ymin, ymax).mean(dim='time'), 'mean speed')
mask_clip = clip(mask, xmin, xmax, ymin, ymax)
speed = speed.where(mask_clip.interp_like(speed, method='nearest') > 0.5)
vmax = max(float(np.nanpercentile(speed.values, 98)), 50.0)

fig, ax = plt.subplots(figsize=(10, 12))
ax.set_facecolor('#d0e8f5')

# hillshade
surf = clip(gimp, xmin, xmax, ymin, ymax)
sv = surf.values
ok = np.where(np.isfinite(sv), sv > 1.0, False)
shade = LightSource(azdeg=315, altdeg=45).hillshade(np.where(ok, sv, 0.0), vert_exag=10, dx=150, dy=150)
rgba = np.zeros((shade.shape[0], shade.shape[1], 4), dtype=np.float32)
rgba[..., 0] = shade
rgba[..., 1] = shade
rgba[..., 2] = shade
rgba[..., 3] = np.where(ok, 0.60, 0.0)
surf_extent = [float(surf.x.min()), float(surf.x.max()), float(surf.y.min()), float(surf.y.max())]
ax.imshow(rgba, extent=surf_extent, origin='upper', zorder=1)

ax.pcolormesh(mask_clip.x, mask_clip.y, xr.where(mask_clip > 0.5, 1.0, np.nan),
              cmap=mcolors.ListedColormap(['white']), vmin=0, vmax=2, alpha=0.75, zorder=2)
im = ax.pcolormesh(speed.x, speed.y, speed, cmap='YlGnBu', norm=mcolors.Normalize(vmin=0.0, vmax=vmax),
                   alpha=0.82, zorder=3)
ax.contour(mask_clip.x, mask_clip.y, mask_clip, levels=[0.5], colors='black', linewidths=1.0, zorder=6)

gate_labels = sorted(gate_pixels)
tab10 = plt.cm.tab10(np.linspace(0, 1, len(gate_labels)))
colours = {}
for k in range(len(gate_labels)):
    colours[gate_labels[k]] = tab10[k]

for i, r in rois.iterrows():
    lbl = int(r['label'])
    roi_x, roi_y = r.geometry.exterior.xy
    gate_x, gate_y = gate_lines[lbl].xy
    ax.plot(roi_x, roi_y, color='crimson', lw=1.8, zorder=8)
    ax.plot(gate_x, gate_y, color=colours[lbl], lw=3.0, solid_capstyle='round', label=f'ROI {lbl}', zorder=9)
    ax.text(r.geometry.centroid.x, r.geometry.centroid.y, str(lbl), fontsize=9, fontweight='bold',
            ha='center', va='center', color=colours[lbl],
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1), zorder=10)

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect('equal')
ax.set_xlabel('Easting (m, EPSG:3413)')
ax.set_ylabel('Northing (m, EPSG:3413)')
ax.set_title('Manual flux gates over mean ice velocity')
ax.legend(ncol=3, fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3, lw=0.5)
cb = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.04, fraction=0.045)
cb.set_label('Mean ice speed (m yr$^{-1}$)')
gate_fig = os.path.join(FIGURE_DIR, 'extra-fig_flux_gates_map.png')
fig.savefig(gate_fig, dpi=150, bbox_inches='tight')
plt.close(fig)


# thinning (Khan et al m/month)
thinning = xr.open_dataset(THINNING_PATH, chunks={'time': 12, 'x': 500, 'y': 500})['dhdt_vol']
thin_time = pd.to_datetime(thinning.time.values)

ds_times = pd.to_datetime(ds_vel.time.values)
start_idx = int(np.argmin(np.abs(thin_time - pd.Timestamp('2007-07-01'))))


# discharge for each gate
all_dfs = {}
H_time_dict = {}
H_fixed_dict = {}

for lbl in sorted(gate_pixels):
    pixels = gate_pixels[lbl]
    az = gate_az[lbl]
    print('ROI', lbl)
    gc.collect()
    n_pix = len(pixels)
    xg = [p[0] for p in pixels]
    yg = [p[1] for p in pixels]

    # thickness = GIMP 2007 + Khan thinning
    p10 = 10_000.0
    thin_clip = clip(thinning, min(xg) - p10, max(xg) + p10, min(yg) - p10, max(yg) + p10).load()
    H_clip = clip(H_2007, min(xg) - p10, max(xg) + p10, min(yg) - p10, max(yg) + p10).load()

    H_time = np.zeros((len(ds_times), n_pix))
    for i in range(n_pix):
        px = pixels[i][0]
        py = pixels[i][1]
        h0 = float(H_clip.interp(x=px, y=py, method='nearest'))
        if not np.isfinite(h0):
            h0 = 0.0
        H_time[:, i] = h0

        dhdt = thin_clip.interp(x=px, y=py, method='nearest').values
        valid = np.isfinite(dhdt)
        if not valid.any():
            continue

        # thinning only
        dhdt_neg = np.where(valid & (dhdt < 0), dhdt, 0.0)
        last = np.where(valid)[0][-1]
        rate = min(float(np.nanmean(dhdt[max(0, last - 35):last + 1])), 0.0)
        t_last = thin_time[last]
        for j in range(len(ds_times)):
            t = ds_times[j]
            if t.year < 2007:
                cum = 0.0
            elif t <= t_last:
                ti = int(np.argmin(np.abs(thin_time - t)))
                cum = float(np.nansum(dhdt_neg[start_idx:ti + 1]))
            else:
                cum = float(np.nansum(dhdt_neg[start_idx:last + 1]))
                months_past = (t - t_last).days / 30.436875
                if months_past > 0:
                    cum += months_past * rate
            H_time[j, i] = h0 + cum

    H_time = np.clip(H_time, 0.0, 5000.0)

    # Bedmachine thickness, fixed in time
    H_fixed = np.zeros(n_pix)
    for i in range(n_pix):
        val = float(H_bm.interp(x=pixels[i][0], y=pixels[i][1], method='nearest').values)
        if np.isfinite(val):
            H_fixed[i] = val
    H_fixed = np.clip(H_fixed, 0.0, 5000.0)
    H_fixed_dict[lbl] = H_fixed
    H_time_dict[lbl] = H_time

    # thickness used for discharge
    if FIXED_THICKNESS:
        H_use = np.tile(H_fixed, (len(ds_times), 1))
    else:
        H_use = H_time

    # Bedmachine bed error used as thickness error
    sigH = np.zeros(n_pix)
    for i in range(n_pix):
        val = float(errbed.interp(x=pixels[i][0], y=pixels[i][1], method='nearest').values)
        if np.isfinite(val):
            sigH[i] = val

    # velocity at the gate
    p4 = PIXEL_W * 4
    ds_c = clip(ds_vel, min(xg) - p4, max(xg) + p4, min(yg) - p4, max(yg) + p4)
    v = compute_with_retry(ds_c[[vx_var, vy_var, e_var]], f'ROI {lbl}')
    v = v.interpolate_na(dim='time', method='linear')

    xi = [int(np.argmin(np.abs(ds_c.x.values - gx))) for gx in xg]
    yi = [int(np.argmin(np.abs(ds_c.y.values - gy))) for gy in yg]
    vx = v[vx_var].values[:, yi, xi]   # (time, gate pixel)
    vy = v[vy_var].values[:, yi, xi]
    ev = v[e_var].values[:, yi, xi]
    del v
    gc.collect()

    # fill missing pixels with gate mean
    for t in range(vx.shape[0]):
        obs = np.isfinite(vx[t]) & np.isfinite(vy[t])
        if (~obs).any() and obs.sum() / n_pix >= 0.80:
            vx[t, ~obs] = np.mean(vx[t, obs])
            vy[t, ~obs] = np.mean(vy[t, obs])

    vperp = vx * np.cos(np.radians(az)) - vy * np.sin(np.radians(az))

    records = []
    for t in range(len(ds_times)):
        H = np.where(H_use[t] > 0, H_use[t], 0.0)
        ice = H > 0
        n_ice = int(ice.sum())
        valid = np.isfinite(vperp[t]) & ice
        n_valid = int(valid.sum())
        if n_ice < 2 or n_valid < 1 or n_valid / n_ice < MIN_GATE_COVERAGE:
            continue

        vp = np.maximum(vperp[t][valid], 0.0)   # no backwards flow
        Hv = H[valid]
        sHv = sigH[valid]
        D = RHO_ICE * float(np.sum(vp * Hv)) * PIXEL_W / 1e12   # Gt/yr

        # same pixels with both thicknesses
        D_fixed = RHO_ICE * float(np.sum(vp * H_fixed[valid])) * PIXEL_W / 1e12
        D_varying = RHO_ICE * float(np.sum(vp * H_time[t][valid])) * PIXEL_W / 1e12

        # errors added in quadrature, missing velocity errors = gate median
        e = ev[t][valid]
        has_e = np.isfinite(e) & (e > 0)
        u_err = np.nan
        l_err = np.nan
        if has_e.any():
            sig_v = np.where(has_e, e, float(np.median(e[has_e])))
            err_sum = float(np.sum(Hv**2 * sig_v**2 + vp**2 * sHv**2))
            u_err = float(np.sqrt((RHO_ICE * PIXEL_W)**2 * err_sum)) / 1e12
            l_err = u_err

        records.append({'time': ds_times[t],
                        'discharge_gt_yr': D,
                        'u_error': u_err,
                        'l_error': l_err,
                        'mean_vperp': float(np.mean(vp)),
                        'mean_thickness': float(np.mean(Hv)),
                        'n_pixels_valid': n_valid,
                        'n_pixels_gate': n_pix,
                        'n_pixels_ice': n_ice,
                        'gate_coverage': round(n_valid / n_ice, 2),
                        'discharge_fixed_H': D_fixed,
                        'discharge_varying_H': D_varying})

    df = pd.DataFrame(records).sort_values('time').reset_index(drop=True)
    print('  mean discharge', round(df['discharge_gt_yr'].mean(), 4), 'Gt/yr')
    df.to_csv(os.path.join(OUTPUT_DIR, f'discharge_ROI_{lbl}.csv'), index=False)
    all_dfs[lbl] = df

client.close()
cluster.close()


# gate thickness plot
n = len(H_time_dict)
fig, axes = plt.subplots(n, 1, figsize=(10, 2.2 * n), sharex=True, gridspec_kw={'hspace': 0.08})
k = 0
for roi in H_time_dict:
    ax = axes[k]
    H = H_time_dict[roi]
    ax.plot(ds_times, H[:, H.shape[1] // 2], color='#1f77b4', lw=1.8, alpha=0.99, label='GIMP 2007 + thinning')
    ax.axhline(H_fixed_dict[roi][H.shape[1] // 2], color='#ff7f0e', lw=1.4, ls='--', label='BedMachine (fixed)')
    ax.axvline(EVENT_DATE, color='crimson', lw=0.8, ls='--', alpha=0.7)
    ax.set_ylabel('H (m)', fontsize=8)
    ax.text(0.02, 0.88, f'L{roi}', transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=7)
    k += 1
axes[0].legend(fontsize=7, loc='upper right')
axes[-1].set_xlabel('Date', fontsize=9)
fig.text(0.005, 0.5, 'Gate ice thickness (m)', va='center', rotation='vertical', fontsize=9)
plt.tight_layout()
out = os.path.join(FIGURE_DIR, 'extra-fig_gate_thickness_timeseries.png')
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.close(fig)


# discharge plot
rois_sorted = sorted(SELECTED_ROI_LABELS)
fig, axes = plt.subplots(len(rois_sorted), 1, figsize=(8.0, len(rois_sorted) * 1.15), sharex=True,
                         gridspec_kw={'hspace': 0.04})
for k in range(len(rois_sorted)):
    ax = axes[k]
    roi = rois_sorted[k]
    d = all_dfs[roi].sort_values('time').set_index('time')
    d = d.resample('W').agg({'discharge_gt_yr': 'mean', 'u_error': 'mean', 'l_error': 'mean'})
    d = d.dropna(subset=['discharge_gt_yr']).reset_index()
    ax.fill_between(d['time'], d['discharge_gt_yr'] - d['l_error'].fillna(0),
                    d['discharge_gt_yr'] + d['u_error'].fillna(0),
                    color='#4C9BE8', alpha=0.22, linewidth=0, zorder=2)
    ax.plot(d['time'], d['discharge_gt_yr'], color='#1a6fbf', lw=1.0, zorder=3)
    ax.axvline(EVENT_DATE, color='#c0392b', lw=0.9, ls='--', alpha=0.85, zorder=4)
    ax.text(0.013, 0.84, f'L{roi}', transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')
    ax.tick_params(labelsize=7)
    ax.set_xlim(PLOT_XMIN, PLOT_XMAX)
    ax.grid(True, alpha=0.20, lw=0.4)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3, prune='both'))
    ax.spines[['top', 'right']].set_visible(False)
axes[-1].set_xlabel('Date', fontsize=9)
fig.text(0.04, 0.5, 'Ice discharge (Gt yr$^{-1}$)', rotation='vertical', va='center', fontsize=9)
fig.tight_layout(rect=[0.07, 0.02, 1.0, 1.0])
p = os.path.join(FIGURE_DIR, 'extra-fig_ice-discharge_supp.png')
fig.savefig(p, dpi=250, bbox_inches='tight')
plt.close(fig)


# fixed vs varying thickness, winter anomalies vs the 2016/17-2020/21 median
post_winters = {'W1': 2022, 'W2': 2023, 'W3': 2024, 'W4': 2025}
comp_rows = []
anom_dfs = {}
for roi in rois_sorted:
    d = all_dfs[roi].copy()
    month = d['time'].dt.month
    winter = np.where(month >= 11, d['time'].dt.year + 1, d['time'].dt.year)
    in_winter = (month >= 11) | (month <= 4)
    base = in_winter & (winter >= 2017) & (winter <= 2021)
    base_fixed = d.loc[base, 'discharge_fixed_H'].median()
    base_varying = d.loc[base, 'discharge_varying_H'].median()
    d['anom_fixed'] = (d['discharge_fixed_H'] - base_fixed) / base_fixed * 100
    d['anom_varying'] = (d['discharge_varying_H'] - base_varying) / base_varying * 100
    anom_dfs[roi] = d

    row = {'roi_label': roi}
    for wn, yr in post_winters.items():
        w = in_winter & (winter == yr)
        a_fix = d.loc[w, 'discharge_fixed_H'].median() / base_fixed * 100 - 100
        a_var = d.loc[w, 'discharge_varying_H'].median() / base_varying * 100 - 100
        row[f'{wn}_anom_fixed'] = a_fix
        row[f'{wn}_anom_varying'] = a_var
        row[f'{wn}_diff'] = a_var - a_fix
    comp_rows.append(row)
comp = pd.DataFrame(comp_rows)
comp.to_csv(os.path.join(OUTPUT_DIR, 'discharge_thickness_comparison.csv'), index=False)
print(comp[['roi_label', 'W1_diff', 'W2_diff', 'W3_diff', 'W4_diff']].round(2))

fig, axes = plt.subplots(len(rois_sorted), 1, figsize=(8.0, len(rois_sorted) * 1.15), sharex=True,
                         gridspec_kw={'hspace': 0.04})
for k in range(len(rois_sorted)):
    ax = axes[k]
    roi = rois_sorted[k]
    d = anom_dfs[roi].set_index('time')[['anom_fixed', 'anom_varying']].resample('W').mean().dropna()
    ax.plot(d.index, d['anom_fixed'], color='#ff7f0e', lw=1.0, label='BedMachine (fixed)')
    ax.plot(d.index, d['anom_varying'], color='#1a6fbf', lw=1.0, label='GIMP 2007 + thinning')
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(EVENT_DATE, color='#c0392b', lw=0.9, ls='--', alpha=0.85)
    ax.text(0.013, 0.84, f'L{roi}', transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')
    ax.tick_params(labelsize=7)
    ax.set_xlim(PLOT_XMIN, PLOT_XMAX)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3, prune='both'))
    ax.spines[['top', 'right']].set_visible(False)
axes[0].legend(fontsize=7, loc='upper right', ncol=2, frameon=False)
axes[-1].set_xlabel('Date', fontsize=9)
fig.text(0.04, 0.5, 'Discharge anomaly (% of baseline)', rotation='vertical', va='center', fontsize=9)
fig.tight_layout(rect=[0.07, 0.02, 1.0, 1.0])
p = os.path.join(FIGURE_DIR, 'extra-fig_discharge_thickness_comparison.png')
fig.savefig(p, dpi=250, bbox_inches='tight')
plt.close(fig)
print('done')
