import xarray as xr
import intake
import pandas as pd
from collections import defaultdict
import dask
import fsspec
import xarray as xr
import numpy as np

def calcMean(data):

	#unfortunatly not all models use lat/lon naming... fixing this
	if ('latitude' in data.dims):
		data = data.rename({'latitude': 'lat'})

	weights = np.cos(np.deg2rad(data.lat))
	data_weighted = data.weighted(weights)
	return data_weighted.mean(("lon", "lat"),skipna=True)

def drop_all_bounds(ds):
	drop_vars = [vname for vname in ds.coords
		if (('_bounds') in vname ) or ('_bnds') in vname]
	return ds.drop(drop_vars)

def open_dset(df):
	assert len(df) == 1
	ds = xr.open_zarr(fsspec.get_mapper(df.zstore.values[0]), consolidated=True)
	return drop_all_bounds(ds)

def open_delayed(df):
	return dask.delayed(open_dset)(df)


	
models=pd.read_csv("model_list_cleaned.csv")
exps=models.scenario.unique()
dsall_tas = {}
for exp in exps:
	dsall_tas[exp] = {}
dsall_tas["historical"] = {}

cat_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
col = intake.open_esm_datastore(cat_url)
time = pd.date_range(start="1850-01-01", end="2099-12-31", freq="MS")  # 'MS' = Month Start


for index, (df_index,item) in enumerate(models.iterrows(), start=0):
	print(item)
	cat = col.search(experiment_id=['historical',item.scenario], source_id=item.model,member_id=item.member,table_id='Amon', variable_id='tas')
	#print(cat)
	ds=cat.to_dataset_dict(zarr_kwargs={'consolidated': True}, decode_times=False)
	key=list(ds.keys())
	ds=xr.concat([ds[key[1]],ds[key[0]]],dim='time').isel(time=slice(0,3000))
	print(len(time),len(ds.time))
	ds = ds.assign_coords(time=time)
	ds=calcMean(ds)
	print(ds)
	dsall_tas[item.scenario][item.model] = ds-ds.sel(time=slice('1850-01-01','1901-01-01')).mean('time')
	

