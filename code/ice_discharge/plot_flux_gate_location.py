# Map of each ROI and its flux gate

import os
import sys
import glob
import warnings
import numpy as np
import xarray as xr
import rioxarray
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.colors import LightSource
from pyproj import Transformer

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import config

VELOCITY_DATA_DIR = config.VELOCITY_DATA_DIR
BEDMACHINE_PATH = config.BEDMACHINE_PATH
MASK_PATH_FILE = config.MASK_PATH_FILE
ROI_SHAPEFILE_PATH = config.ROI_SHAPEFILE_PATH
MANUAL_GATES_PATH = os.path.join(config.DATA_DIR, "ice_discharge", "flux_gates_manual.gpkg")
FIGURE_DIR = config.FIG_DIR
OUT_FIG = os.path.join(FIGURE_DIR, "FigS3_flux_gates_map.png")

SELECTED_ROI_LABELS = config.ROI_LABELS
TARGET_YEAR = 2018
MAP_PAD_M = 18_000.0
ROI_COLOR = "crimson"
REGION_BOX = {"minx": -700_000.0, "miny": -3_400_000.0, "maxx": 900_000.0, "maxy": -600_000.0}


# clip to a box, y can be ascending or descending
def clip(da, xmin, xmax, ymin, ymax):
    da = da.sel(x=slice(xmin, xmax))
    if float(da.y[0]) > float(da.y[-1]):
        return da.sel(y=slice(ymax, ymin))
    return da.sel(y=slice(ymin, ymax))


# e.g. 50.25°W
def deg(val, pos_dir, neg_dir):
    if val > 0:
        return f"{abs(val):.2f}°{pos_dir}"
    return f"{abs(val):.2f}°{neg_dir}"


os.makedirs(FIGURE_DIR, exist_ok=True)

rois = gpd.read_file(ROI_SHAPEFILE_PATH).to_crs("EPSG:3413")
rois["id"] = rois["id"].astype(int)
rois = rois[rois["id"].isin(SELECTED_ROI_LABELS)].copy()

gates = gpd.read_file(MANUAL_GATES_PATH).to_crs("EPSG:3413")
gates["roi_label"] = gates["roi_label"].astype(int)

files = sorted(glob.glob(os.path.join(VELOCITY_DATA_DIR, "*.nc")))
ds = xr.open_mfdataset(files, combine="by_coords", parallel=False, chunks={"time": 1}, engine="netcdf4")
ds = ds.drop_vars(["time_bnds", "lat", "lon", "lat_bnds", "lon_bnds"], errors="ignore")
ds = ds.rio.write_crs("EPSG:3413")
ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
ds = ds.rio.clip_box(**REGION_BOX)
v = ds["land_ice_surface_velocity_magnitude"] * 365.25   # m/day -> m/yr
vel = v.sel(time=ds.time.dt.year == TARGET_YEAR).mean(dim="time").compute()

mask_ds = xr.open_dataset(MASK_PATH_FILE).rio.write_crs("EPSG:3413")
mask_var = list(mask_ds.data_vars)[0]
mask = xr.where(mask_ds[mask_var] == 2, 1, 0).squeeze().interp_like(vel, method="nearest")
surface = xr.open_dataset(BEDMACHINE_PATH)["surface"].squeeze().interp_like(vel, method="linear")

# same size window for all panels
roi_ids = sorted(SELECTED_ROI_LABELS)
roi_geoms = {}
for r in roi_ids:
    roi_geoms[r] = rois[rois["id"] == r].iloc[0].geometry
w = max([g.bounds[2] - g.bounds[0] for g in roi_geoms.values()]) + 2 * MAP_PAD_M
h = max([g.bounds[3] - g.bounds[1] for g in roi_geoms.values()]) + 2 * MAP_PAD_M

windows = {}
vals = []
for r in roi_ids:
    g = roi_geoms[r]
    cx = (g.bounds[0] + g.bounds[2]) / 2
    cy = (g.bounds[1] + g.bounds[3]) / 2
    windows[r] = (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)
    vv = clip(vel, cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2).values
    vals.append(vv[np.isfinite(vv)])
vmax = max(float(np.percentile(np.concatenate(vals), 98)), 50.0)

to_lonlat = Transformer.from_crs("EPSG:3413", "EPSG:4326", always_xy=True)
fig, axes = plt.subplots(2, 5, figsize=(8.0, 4.0), layout="compressed")
axes = axes.flatten()

for i in range(len(roi_ids)):
    roi_id = roi_ids[i]
    ax = axes[i]
    roi = roi_geoms[roi_id]
    xmin, xmax, ymin, ymax = windows[roi_id]
    ax.set_facecolor("#AFCED9")

    vel_c = clip(vel, xmin, xmax, ymin, ymax)
    mask_c = clip(mask, xmin, xmax, ymin, ymax)
    surf_c = clip(surface, xmin, xmax, ymin, ymax)

    # grey land + hillshade
    sv = surf_c.values
    land = np.where(np.isfinite(sv), sv > 1.0, False)
    shade = LightSource(azdeg=315, altdeg=45).hillshade(np.where(land, sv, 0.0), vert_exag=10, dx=150, dy=150)
    ext = [float(surf_c.x.min()), float(surf_c.x.max()), float(surf_c.y.min()), float(surf_c.y.max())]
    grey = mcolors.to_rgba("#dfdfdf")
    base = np.zeros((shade.shape[0], shade.shape[1], 4), dtype=np.float32)
    base[..., 0] = grey[0]
    base[..., 1] = grey[1]
    base[..., 2] = grey[2]
    base[..., 3] = np.where(land, 1.0, 0.0)
    ax.imshow(base, extent=ext, origin="upper", zorder=0.5)
    hs = np.zeros((shade.shape[0], shade.shape[1], 4), dtype=np.float32)
    hs[..., 0] = shade
    hs[..., 1] = shade
    hs[..., 2] = shade
    hs[..., 3] = np.where(land, 0.60, 0.0)
    ax.imshow(hs, extent=ext, origin="upper", zorder=1)

    ice = xr.where(mask_c > 0.5, 1.0, np.nan)
    ax.pcolormesh(ice.x, ice.y, ice, cmap=mcolors.ListedColormap(["white"]), vmin=0, vmax=2, alpha=0.75, zorder=2)
    pc = ax.pcolormesh(vel_c.x, vel_c.y, vel_c, cmap="YlGnBu", norm=mcolors.Normalize(vmin=0, vmax=vmax),
                       alpha=0.82, zorder=3)
    ax.contour(mask_c.x, mask_c.y, mask_c, levels=[0.5], colors="black", linewidths=1.0, zorder=6)

    roi_x, roi_y = roi.exterior.xy
    ax.plot(roi_x, roi_y, color=ROI_COLOR, linewidth=2.3, zorder=8)

    # flux gate
    gate = gates[gates["roi_label"] == roi_id].unary_union
    if gate.geom_type == "MultiLineString":
        segments = list(gate.geoms)
    else:
        segments = [gate]
    for seg in segments:
        seg_x, seg_y = seg.xy
        ax.plot(seg_x, seg_y, color="black", linewidth=1.5, zorder=9)

    # label
    cx = roi.centroid.x
    cy = roi.centroid.y
    tx = cx + (xmax - xmin) * 0.25
    ty = cy - (ymax - ymin) * 0.25
    ax.plot(cx, cy, marker="o", color=ROI_COLOR, markersize=5, zorder=10)
    ax.plot([cx, tx], [cy, ty], color=ROI_COLOR, linewidth=1.2, zorder=9)
    ax.text(tx, ty, f"L{roi_id}", fontsize=11, fontweight="bold", ha="center", va="center", color="black",
            zorder=10, path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")

    # lat/lon ticks, two per axis
    xmid = (xmin + xmax) / 2
    ymid = (ymin + ymax) / 2
    xticks = np.linspace(xmin + 4000, xmax - 4000, 2)
    yticks = np.linspace(ymin + 4000, ymax - 4000, 2)
    xlabels = []
    for xt in xticks:
        lon, lat = to_lonlat.transform(xt, ymid)
        xlabels.append(deg(lon, "E", "W"))
    ylabels = []
    for yt in yticks:
        lon, lat = to_lonlat.transform(xmid, yt)
        ylabels.append(deg(lat, "N", "S"))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, left=True, labelleft=True,
                   right=False, labelright=False, length=2.5, width=0.7, pad=1.0, labelsize=7)
    ax.yaxis.set_tick_params(labelrotation=90)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333333")

    letter = "abcdefghij"[i]
    ax.text(0.06, 0.94, f"({letter})", transform=ax.transAxes, fontsize=13, fontweight="bold",
            va="top", ha="left", path_effects=[pe.withStroke(linewidth=2.5, foreground="white")], zorder=15)

cbar = fig.colorbar(pc, ax=axes, orientation="horizontal", shrink=0.55, pad=0.03, aspect=35)
cbar.set_label("Mean annual ice velocity (m yr$^{-1}$)", fontsize=11)
cbar.ax.tick_params(labelsize=9)

fig.savefig(OUT_FIG, dpi=300)
plt.close(fig)
