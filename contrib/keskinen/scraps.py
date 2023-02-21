# # from rioxarray.merge import merge_arrays

# # def merge_s1_times(dataset, times, verbose = False):
# #     if verbose:
# #         print('Merging:', times)
# #     das = [dataset.sel(time = ts)['s1'] for ts in times]
# #     assert len(das) == len([d for d in das if d['relative_orbit'] == das[0]['relative_orbit']])
# #     print(das[0].where(das[0] == 0).notnull().sum())
# #     if das[0].where(das[0] == 0).notnull().sum() > 0:
# #         nodata_value = 0
# #     else:
# #         nodata_value = np.nan
# #     print(nodata_value)
# #     merged = merge_arrays(das, crs = 'EPSG:4326', nodata= nodata_value)
# #     dataset = dataset.drop_sel(time = times[1:])
# #     dataset['s1'].loc[dict(time = times[0])] = merged.values
# #     times = []
# #     return dataset, times, das

# # import pandas as pd

# # def s1_partial_merge(dataset):
# #     times = []
# #     for ts in dataset.time.values:
        
# #         if not times:
# #             times.append(ts)
# #             continue

# #         if ts - times[0] > pd.Timedelta('1 minute'):
# #             if len(times) == 1:
# #                 times = [ts]
# #                 continue
# #             else:
# #                 dataset, times, das = merge_s1_times(dataset, times)
# #         else:
# #             times.append(ts)
        
# #         if ts == ds.time.values[-1] and len(times) > 1:
# #             dataset, times, das = merge_s1_times(dataset, times)
        
# #     return dataset, das


# def reduce_snow_index_snowfree(dataset: xr.Dataset) -> xr.Dataset: 
#     """
#     Reduce increment if prior date was still snowfree (to avoid jumps in shallow snow)
    
#     # change in cross-pol and VV is reduced by 0.5 when past IMS snow cover is snow-free
#     dra_u(sc_pri==0)=0.5.*dra_u(sc_pri==0);
#     dvv_u(sc_pri==0)=0.5.*dvv_u(sc_pri==0);
    
#     Args:
#     dataset: Xarray Dataset of sentinel images with delta-gamma

#     Returns:
#     dataset: Xarray Dataset of sentinel images with 2.5 +  reduced
#     for snow-free times
#     """
#     pass

# def reduce_snow_index_big_decreases(dataset: xr.Dataset) -> xr.Dataset: 
#     """
#     Reduce impact of strong decreases in backscatter 
#     (e.g. caused by rain on snow, or wet snow)
    
#     # change in cross-pol and VV when change is decreasing more than 2.5
#     # if d_vv or d_cross < -2.5 then d_XX = -2.5 + 0.5 * (2.5 + d_XX)
#     dra_u(dra_u<-2.5)=-2.5+0.5.*(2.5+dra_u(dra_u<-2.5));  
#     dvv_u(dvv_u<-2.5)=-2.5+0.5.*(2.5+dvv_u(dvv_u<-2.5));  

#     Args:
#     dataset: Xarray Dataset of sentinel images with delta-gamma

#     Returns:
#     dataset: Xarray Dataset of sentinel images with big decreases mitigated
#     """

#     pass

# def reduce_snow_index_big_increases(dataset: xr.Dataset) -> xr.Dataset: 
#     """
#     Reduce impact of strong increases in backscatter 
#     (e.g. caused by (re)frost, or wet snow roughness)
    
#     # change in cross-pol and VV when change is decreasing more than 2.5
#     # if d_vv or d_cross > 2.5 then d_XX = 2.5 + 0.5 * ( d_XX - 2.5)
#     dra_u(dra_u>2.5)=2.5+0.5.*(dra_u(dra_u>2.5)-2.5);
#     dvv_u(dvv_u>2.5)=2.5+0.5.*(dvv_u(dvv_u>2.5)-2.5);

#     Args:
#     dataset: Xarray Dataset of sentinel images with delta-gamma

#     Returns:
#     dataset: Xarray Dataset of sentinel images with big increases mitigated
#     """

#     pass

# def calc_timesteps_weights(dataset: xr.Dataset) -> np.array:
#     """
#     Calculate weight for snow index effects of surrounding snow indexes

#     Matlab code: 
#     # make initial weights
#     wgts=repmat(win+1-abs([-win:win]),dim,1); [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]
#     # set no weights to steps steps without obs
#     wgts(isnan(si_tmp))=NaN;
#     # Reduce weight (e.g. by 3) for time steps with wet snow
#     wgts(wet_tmp==1)=wgts(wet_tmp==1)./3;

#     Args:
#     dataset: Xarray Dataset of sentinel images with snow-index

#     Returns:
#     weights: weights for 5, 11 days
#     """
#     pass

# def add_band(dataset: xr.Dataset, dataArray: xr.DataArray, band_name: str, inplace: bool = False) -> xr.Dataset:
#     """
#     Add band to sentinel 1 datavariable in Dataset. The dataArray to add can have
#     multiple time steps and already exists.

#     Args:
#     dataset: Dataset containing sentinel-1 images
#     dataArray: dataArray to add to dataset
#     band_name: name for band, can already be in dataset
#     inplace: operate on dataset in place or return copy

#     Returns:
#     dataset: dataset with dataArray add to it at the dataArray's timeslice with
#     that band name.
#     """
#     # check inplace flag
#     if not inplace:
#         dataset = dataset.copy(deep=True)

#     # remove DataArrays coordinates so they don't concatenated only band dimension
#     dataArray = dataArray.reset_coords(names = ['flight_dir', 'platform', 'relative_orbit'], drop = True)
    
#     # rename dataArrays band
#     dataArray['band'] = band_name

#     # if this is the first time adding this band
#     if band_name not in dataset.band:
#         # combine the 's1' dataArray and your new data on band
#         com_da = xr.concat([dataset['s1'], dataArray], dim = 'band')

#         # remove s1 from data variables to remerge
#         vars = list(dataset.data_vars)
#         vars.remove('s1')

#         # merge dataset sans sentinel1 with sentinel dataArray with new band
#         dataset = xr.merge([dataset[vars], com_da])
#     else:
#         # if band already in dataset just slice into data array time slices and add data
#         dataset['s1'].loc[dict(band = band_name, time = dataArray.time)] = dataArray

#     if not inplace:
        # return dataset