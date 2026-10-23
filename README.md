# GrIS multi-year winter slowdown

Code and figures for the analysis on the multi-year winter slowdown of the western Greenland Ice Sheet after the 14 August 2021 melt event.

Manuscript in preparation for submission to The Cryosphere:

Ing, R. N., Davison, B. J., Nienow, P. W., and Sole, A. J.: Extreme melt event causes multi-year winter slowdown of the western Greenland Ice Sheet, in preparation.

## Layout

```
config.py          paths and study area
code/              scripts
data/external/     external data
data/derived/      data made by the scripts (Zenodo)
figures/           figures
```

## Data

External data used:

- PROMICE Sentinel-1 ice velocity (Solgaard et al., 2021), https://doi.org/10.22008/FK2/K70OPK
- BedMachine Greenland v6 (Morlighem et al., 2017), https://doi.org/10.5067/7U2049INNRMP
- MARv3.14.3, ERA5 and CESM2 SSP3-7.0 forced (Fettweis et al., 2017; Hofer et al., 2020) http://ftp.climato.be/fettweis/MARv3.14/Greenland/CESM2-ssp370-10km-daily/
- Mankoff et al. (2020) runoff, https://doi.org/10.22008/FK2/XKQVL7
- GIMP DEM (Howat et al., 2014), https://doi.org/10.5067/B8X58MQBFUPA
- Khan et al. (2025) thinning, https://datadryad.org/dataset/doi:10.5061/dryad.s4mw6m9dh
- ERA5 (Copernicus Climate Data Store), in `data/external/era5/`
- Watson River discharge (GEUS), https://doi.org/10.22008/FK2/XEHYCM, into `data/external/watson/`
- GrIML ice-marginal lakes (How et al., 2025), https://doi.org/10.22008/FK2/MBKW9N, into `data/external/ice_marginal_lake/`

Derived data produced by these scripts (ice discharge, velocity timeseries) is available from (), 

## Scripts

- `velocity/`: `prepare_land-term_velocity_data.py`, `create_select_winters_tool.py` (to manually select winter windows) and `prepare_mtg_velocity_data.py`, then run plots
- `scatter_plots/`: `prepare_vel_scatter_data.py` and `prep_flowline_data.py`, then run plotting scripts
- `met/`: melt event, return periods and Watson River plots. `fig_return_period_runoff_increase.py` 
- `ice_discharge/`: `calculate_ice_discharge.py`, then `calculate_ice_discharge_deficit.py` and the gate map
- `subglacial/`: cavity recovery and channel closure
- `ice_marginal_lake/`: lake drainage near M3
