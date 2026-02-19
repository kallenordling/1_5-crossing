import xarray as xr
import numpy as np
models=['CanESM5','MIROC6','ACCESS-ESM1-5']
ssps=['ssp126','ssp245','ssp370','ssp585']
for model in models:
    for ssp in ssps:
        ds_ssp=xr.open_dataset('gwl_data/'+model+"_"+ssp+"_global_mean_2015-2100.nc")
        ds_hist=xr.open_dataset('gwl_data/'+model+"_historical_global_mean_1850-2014.nc")
        common_members = np.intersect1d(ds_ssp.ensemble_member, ds_hist.ensemble_member)
        ds_ssp = ds_ssp.sel(ensemble_member=common_members)
        ds_hist = ds_hist.sel(ensemble_member=common_members)
        ds_combined = xr.concat([ds_hist, ds_ssp], dim="time")
        ds_combined = ds_combined - ds_combined.sel(time=slice('1850-01-01','1901-01-01')).mean('time')
        ds_combined = ds_combined.rename({"ensemble_member": "member_id"})
        ds_combined.to_netcdf('cmip6/'+ssp+"_"+model+".nc")
        print(ds_combined)