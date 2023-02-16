# from rioxarray.merge import merge_arrays

# def merge_s1_times(dataset, times, verbose = False):
#     if verbose:
#         print('Merging:', times)
#     das = [dataset.sel(time = ts)['s1'] for ts in times]
#     assert len(das) == len([d for d in das if d['relative_orbit'] == das[0]['relative_orbit']])
#     print(das[0].where(das[0] == 0).notnull().sum())
#     if das[0].where(das[0] == 0).notnull().sum() > 0:
#         nodata_value = 0
#     else:
#         nodata_value = np.nan
#     print(nodata_value)
#     merged = merge_arrays(das, crs = 'EPSG:4326', nodata= nodata_value)
#     dataset = dataset.drop_sel(time = times[1:])
#     dataset['s1'].loc[dict(time = times[0])] = merged.values
#     times = []
#     return dataset, times, das

# import pandas as pd

# def s1_partial_merge(dataset):
#     times = []
#     for ts in dataset.time.values:
        
#         if not times:
#             times.append(ts)
#             continue

#         if ts - times[0] > pd.Timedelta('1 minute'):
#             if len(times) == 1:
#                 times = [ts]
#                 continue
#             else:
#                 dataset, times, das = merge_s1_times(dataset, times)
#         else:
#             times.append(ts)
        
#         if ts == ds.time.values[-1] and len(times) > 1:
#             dataset, times, das = merge_s1_times(dataset, times)
        
#     return dataset, das