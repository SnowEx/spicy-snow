import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error

res_dir = Path('/bsuhome/zacharykeskinen/spicy-snow/results/site_param_rmses/')

dfs = {}
for res in res_dir.glob('*.csv'):
    df = pd.read_csv(res, index_col = 0)
    dfs[res.stem] = df

fig, axes = plt.subplots(3, 1, figsize = (12, 8))

for i, (col1, col2) in enumerate([['varying_all', 'varying_c'], ['varying_all', 'all_sites'], ['varying_all','leivens_params']]):
    diff = dfs[col2]['rmse'] - dfs[col1]['rmse']
    param_index = dfs['varying_all'].index + '\n a:' +dfs[col1].a.astype(str) + ' b:' +dfs[col1].b.astype(str) + ' c:' +dfs[col1].c.astype(str) + '\n' + ' a:' +dfs[col2].a.astype(str) + ' b:' +dfs[col2].b.astype(str) + ' c:' +dfs[col2].c.astype(str)
    diff = diff.set_axis(param_index.values)
    diff.plot(linestyle = 'None', marker = 'x', ax = axes[i])

axes[0].set_title('RMSE site-by-site optimization - RMSE site-by-site optimization only varying C')
axes[1].set_title('RMSE site-by-site optimization - RMSE optimizing for all data')
axes[2].set_title('RMSE site-by-site optimization - RMSE Leivens et al. (2021)')

for ax in axes:
    ax.set_ylabel('RMSE Difference')
    ax.set_ylim(0, 0.12)
    ax.tick_params(axis='x', labelrotation = 0, labelsize = 7)

plt.tight_layout()
plt.draw()
