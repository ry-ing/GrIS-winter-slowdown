# GrIS multi-year winter slowdown

Code and figures manuscript:

Ing, R. N., Davison, B. J., Nienow, P. W., and Sole, A. J.: Extreme melt event causes multi-year winter slowdown of the western Greenland Ice Sheet, in preparation.

## Layout

```
config.py          paths and study area
code/              scripts
data/external/     external data
data/derived/      data made by the scripts 
figures/           figures
```

## Data

External data used:

- PROMICE Sentinel-1 ice velocity (Solgaard et al., 2021), https://doi.org/10.22008/FK2/K70OPK
- BedMachine Greenland v6 (Morlighem et al., 2017), https://doi.org/10.5067/7U2049INNRMP
- MARv3.14.3, ERA5 and CESM2 SSP3-7.0 forced (Fettweis et al., 2017; Hofer et al., 2020) http://ftp.climato.be/fettweis/MARv3.14/Greenland/CESM2-ssp370-10km-daily/
- Mankoff et al. (2020) runoff, https://doi.org/10.22008/FK2/XKQVL7
- GIMP DEM (Howat et al., 2014), https://doi.org/10.5067/B8X58MQBFUPA
- ERA5 (Copernicus Climate Data Store), in `data/external/era5/`
- Watson River discharge (GEUS), https://doi.org/10.22008/FK2/XEHYCM, into `data/external/watson/`
- GrIML ice-marginal lakes (How et al., 2025), https://doi.org/10.22008/FK2/MBKW9N, into `data/external/ice_marginal_lake/`

Derived data produced by these scripts (ice discharge, velocity timeseries) is available from https://doi.org/10.5281/zenodo.24544148

