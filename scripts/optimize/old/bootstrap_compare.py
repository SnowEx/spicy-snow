# from itertools import product
from tqdm.contrib.itertools import product
import numpy as np
from numpy import load
import pandas as pd
import xarray as xr
from pathlib import Path
from tqdm import tqdm
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

dfs = []
for fp in Path('./param_pdf/').glob('*.csv'):
# for fp in Path('./data/').glob('*.csv'):
    dfs.append(pd.read_csv(fp))


full = pd.concat(dfs, ignore_index = True).drop('Unnamed: 0', axis = 1)

full.drop(['rmse'], axis = 1).plot.kde()
plt.yscale('log')
plt.show()

fig, axes = plt.subplots(3, figsize = (12, 8))

full.drop(['rmse'], axis = 1).plot.kde(ax = axes[0])
axes[0].set_ylabel('')
axes[0].set_yticks([])

ax = axes[1]
full.plot.scatter(x = 'a', y = 'c', ax = ax, alpha = 0.01, color = 'black')

# linear model
x = np.arange(1, 3.5, 0.1)
for _ in range(10000):
    boot = full.sample(len(full), replace = True)
    slope, intercept, r_value, p_value, std_err = stats.linregress(boot.a, boot.c)
    y = slope * x + intercept
    confidence_interval = 2.58*std_err
    ax.plot(x, y, color = 'blue', alpha = 0.01)
ax.set_xticks(np.arange(1, 3.5, 0.5))

sns.violinplot(x = 'a', y = 'c', data = full, ax= axes[2])

plt.suptitle('Summary of 8000 Bootstrapped Parameter Optimizations')

plt.tight_layout()

plt.show()

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

fig = plt.figure(figsize = (12,9))
ax = fig.add_subplot(projection='3d')

sc = ax.scatter(full.a, full.b, full.c, c = full.rmse, marker= 'o', cmap = 'magma')

# https://gist.github.com/aricooperdavis/c658fc1c5d9bdc5b50ec94602328073b
X_train = full[['a','b']].values
y_train = full.c.values
model = sklearn.linear_model.LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_train)
coefs = model.coef_
intercept = model.intercept_
xs = np.tile(np.linspace(min(full.a), max(full.a), 5), (5,1))
ys = np.tile(np.linspace(min(full.b), max(full.b), 5), (5,1)).T
zs = xs*coefs[0]+ys*coefs[1]+intercept
ax.plot_surface(xs,ys,zs, alpha=0.5)

ax.set_xlabel('A')
ax.set_ylabel('B')
ax.set_zlabel('C')

plt.colorbar(sc, orientation = 'horizontal', shrink = 0.4, label = 'RMSE')

plt.title("Equation: c = {:.2f} + {:.2f} * a + {:.2f} * b".format(intercept, coefs[0],
                                                          coefs[1]))

plt.tight_layout()
plt.show()