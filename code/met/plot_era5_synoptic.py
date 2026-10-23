# ERA5 2 m temperature anomaly and MSLP on 14 Aug 2021

import os
import sys
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import config

ERA5_DIR = os.path.join(config.EXTERNAL_DIR, 'era5')

ds_t2m = xr.open_dataset(os.path.join(ERA5_DIR, 't2m_daily_20210814.nc'))
ds_mslp = xr.open_dataset(os.path.join(ERA5_DIR, 'mslp_daily_20210814.nc'))
ds_clim = xr.open_dataset(os.path.join(ERA5_DIR, 't2m_monthly_1980_2010.nc'))

# August mean 1980-2010, deg C
clim_aug = ds_clim['t2m'].sel(valid_time=ds_clim['valid_time'].dt.month == 8).mean(dim='valid_time') - 273.15

# lon 0-360 -> -180-180
lon = ds_t2m.longitude.values
ds_t2m = ds_t2m.assign_coords(longitude=np.where(lon > 180, lon - 360, lon)).sortby('longitude')
lon = ds_mslp.longitude.values
ds_mslp = ds_mslp.assign_coords(longitude=np.where(lon > 180, lon - 360, lon)).sortby('longitude')

# a wide area so the corners of the map are filled
t2m_2021 = ds_t2m['t2m'].isel(valid_time=0).sel(latitude=slice(90, 30), longitude=slice(-140, 30)) - 273.15
msl = ds_mslp['msl'].isel(valid_time=0).sel(latitude=slice(90, 30), longitude=slice(-140, 30)) / 100.0   # hPa

# temperature anomaly
clim_on_grid = clim_aug.interp(latitude=t2m_2021.latitude, longitude=t2m_2021.longitude)
anom = t2m_2021 - clim_on_grid

lon2d, lat2d = np.meshgrid(anom.longitude.values, anom.latitude.values)
lon2d_msl, lat2d_msl = np.meshgrid(msl.longitude.values, msl.latitude.values)
msl_smooth = gaussian_filter(msl.values, sigma=1.5)

# north polar stereographic centred on Greenland
LON0 = -90
LON1 = -18
LAT0 = 40
LAT1 = 85
proj = ccrs.NorthPolarStereo(central_longitude=-54)
data_crs = ccrs.PlateCarree()

# map extent
n = 300
edge_lons = np.concatenate([np.linspace(LON0, LON1, n), np.full(n, LON1),
                            np.linspace(LON1, LON0, n), np.full(n, LON0)])
edge_lats = np.concatenate([np.full(n, LAT0), np.linspace(LAT0, LAT1, n),
                            np.full(n, LAT1), np.linspace(LAT1, LAT0, n)])
edge_xy = proj.transform_points(data_crs, edge_lons, edge_lats)
x0 = float(edge_xy[:, 0].min())
x1 = float(edge_xy[:, 0].max())
y0 = float(edge_xy[:, 1].min())
y1 = float(edge_xy[:, 1].max())

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'axes.linewidth': 0.8,
})

fig, ax = plt.subplots(figsize=(8, 7.5), subplot_kw={'projection': proj}, dpi=150, layout='constrained')
ax.set_extent([x0, x1, y0, y1], crs=proj)

ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '50m', facecolor='#cce0f0', edgecolor='none'),
               zorder=0)

# temperature anomaly
anom_max = max(np.ceil(np.nanpercentile(np.abs(anom.values), 99)), 4.0)
a_levels = np.arange(-anom_max, anom_max + 0.5, 0.5)
cf = ax.contourf(lon2d, lat2d, anom.values, levels=a_levels, cmap='RdBu_r', transform=data_crs,
                 extend='both', zorder=1)

# MSLP
p_levels = np.arange(956, 1048, 4)
p_levels_thick = np.arange(956, 1048, 8)
ax.contour(lon2d_msl, lat2d_msl, msl_smooth, levels=p_levels, colors='black', linewidths=0.5,
           transform=data_crs, zorder=4)
cs_thick = ax.contour(lon2d_msl, lat2d_msl, msl_smooth, levels=p_levels_thick, colors='black', linewidths=1.1,
                      transform=data_crs, zorder=5)
labels = ax.clabel(cs_thick, levels=p_levels_thick, inline=True, inline_spacing=5, fmt='%d', fontsize=8.5,
                   use_clabeltext=True)
for txt in labels:
    txt.set_fontweight('bold')
    txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])
    txt.set_clip_on(True)

# coastlines and borders
ax.add_feature(cfeature.NaturalEarthFeature('physical', 'coastline', '50m', facecolor='none', edgecolor='#1a1a1a'),
               linewidth=0.65, zorder=6)
ax.add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land', '50m', facecolor='none',
                                            edgecolor='#555555'),
               linewidth=0.35, linestyle='--', zorder=6)

gl = ax.gridlines(crs=data_crs, draw_labels=True, linewidth=0.45, color='gray', alpha=0.6, linestyle='--',
                  x_inline=False, y_inline=False, zorder=7)
gl.xlocator = mticker.FixedLocator([-120, -100, -80, -60, -40, -20, 0, 20])
gl.ylocator = mticker.FixedLocator([45, 50, 55, 60, 65, 70, 75, 80, 85])
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 9, 'color': '#222222', 'rotation': 0}
gl.ylabel_style = {'size': 9, 'color': '#222222', 'rotation': 0}

cbar = fig.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, shrink=0.6, extend='both', drawedges=False)
cbar.set_label('2 m Temperature Anomaly (°C)\nrelative to 1980–2010 August mean', fontsize=9.5, labelpad=8)
cbar.ax.tick_params(labelsize=9)
cbar.ax.axhline(0, color='black', linewidth=0.8)

out = os.path.join(config.FIG_DIR, 'Fig02_synoptic_conditions.png')
plt.savefig(out, bbox_inches='tight', dpi=300)
