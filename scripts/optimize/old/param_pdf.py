from itertools import product
# from tqdm.contrib.itertools import product
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import time

# multiprocessing
# https://superfastpython.com/multiprocessing-for-loop/
from multiprocessing import Process

# from itertools import product
from tqdm.contrib.itertools import product
import numpy as np
from numpy import load
import pandas as pd
import xarray as xr
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import time

# multiprocessing
# https://superfastpython.com/multiprocessing-for-loop/
from multiprocessing import Process

def get_bootstrap(x, y):
    # bootstrap
    x_bs = np.random.choice(x, size = len(x))
    y_bs = np.random.choice(y, size = len(y))

    return x_bs, y_bs

def calc_rmse(y_actual, y_pred):
    rms = mean_squared_error(y_actual, y_pred, squared = False)
    return rms

def optimize_params(run_number):

    np.random.seed()

    # Create parameter space
    A = np.round(np.arange(1, 3.1, 0.5), 2)
    B = np.round(np.arange(0, 1.01, 0.1), 2)
    C = np.round(np.arange(0, 1.001, 0.01), 2)
    ABC = [A, B, C]

    iters = 200

    df = pd.DataFrame(np.ones((iters, 4), dtype = float)*10, columns = ['a', 'b', 'c', 'rmse'])

    from itertools import product
    # from tqdm.contrib.itertools import product
    from tqdm import tqdm

    new_param_dir = Path('~/scratch/params_np').expanduser()

    for i in tqdm(range(iters)):
        low_rmse = 1e10
        best_coords = {'a': None, 'b': None, 'c': None}
        for a, b, c in product(*ABC):
            data = load(new_param_dir.joinpath(f'{a}_{b}_{c}.npy'))
                
            rmse = calc_rmse(*get_bootstrap(data[0], data[1])) # row 0 = lidar-sd, 1 = spicy
            if rmse < low_rmse:
                low_rmse = rmse
                best_coords['a'] = a
                best_coords['b'] = b
                best_coords['c'] = c

        df.loc[i, 'a'] = best_coords['a']
        df.loc[i, 'b'] = best_coords['b']
        df.loc[i, 'c'] = best_coords['c']
        df.loc[i, 'rmse'] = float(low_rmse)
    
    out_dir = Path('/bsuhome/zacharykeskinen/spicy-snow/scripts/optimize/param_pdf')
    df.to_csv(out_dir.joinpath(f'run_{run_number}.csv'))

if __name__ == '__main__':
    # create all tasks
    processes = [Process(target=optimize_params, args=(i,)) for i in range(40)]
    # start all processes
    for process in processes:
        process.start()
    # wait for all processes to complete
    for process in processes:
        process.join()
    # report that all tasks are completed
    print('Done', flush=True)