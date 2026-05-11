"""Synthetic snapshot-style end-to-end test for the post-download pipeline.

The download path (S1, VIIRS, Proba-V) is mocked at the boundary by
constructing a realistic Dataset that mimics what would come out of
generate_sentinel1_dataarray + generate_snowcover_dataarray +
generate_forest_fraction_dataarray. The rest of the pipeline
(amplitude_to_dB -> s1_orbit_averaging -> s1_clip_outliers ->
calc_delta_* -> calc_snow_index -> calc_snow_index_to_snow_depth ->
wet_snow flags) runs against it.

This catches a wide class of regressions in the post-download chain
without needing network, Earthdata creds, or real downloads — the gap
that let the bugs from this session leak into a release. The dataset
deliberately includes:

* Multiple orbits with disjoint time sets (catches in-place .loc[] writes)
* One single-acquisition orbit (the T021 shape)
* Multi-timestep snowcover that varies per timestep (catches stale-`ts`)
* Realistic dB-range vv/vh values (catches preprocessing assertions)

It is NOT a numerical-correctness regression test; it asserts the
pipeline runs without error and produces outputs in plausible ranges.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from spicy_snow.processing.s1_preprocessing import (
    amplitude_to_dB, s1_orbit_averaging, s1_clip_outliers,
)
from spicy_snow.processing.snow_index import (
    calc_delta_vv, calc_delta_cross_ratio, calc_delta_gamma,
    clip_delta_gamma_outlier, calc_snow_index, calc_snow_index_to_snow_depth,
)
from spicy_snow.processing.wet_snow import (
    id_newly_wet_snow, id_wet_negative_si, id_newly_frozen_snow, flag_wet_snow,
)


@pytest.fixture
def synthetic_post_download_dataset():
    """Mimics the dataset shape that retrieve_snow_depth has after the
    download + initial assembly step (right before amplitude_to_dB).

    Variables: vv, vh (linear amplitude), snowcover (binary 0/1), fcf
    (0-1), watermask (binary), with a multi-orbit time axis including
    a single-acquisition orbit.
    """
    # 18 timesteps over a winter season, 3 orbits with disjoint time
    # patterns + one orphan single-acquisition orbit (track 21).
    # Sorted chronologically (production data comes through
    # combine_close_images(...sortby('time')) so the snow-index funcs
    # assume a sorted time index for slice-based prev-image lookups).
    times_unsorted = pd.to_datetime([
        # Orbit 10 (6-day)
        '2024-10-01', '2024-10-07', '2024-10-13', '2024-10-19',
        '2024-10-25', '2024-10-31', '2024-11-06',
        # Orbit 65 (6-day, offset by 3 days)
        '2024-10-04', '2024-10-10', '2024-10-16', '2024-10-22',
        '2024-10-28', '2024-11-03', '2024-11-09',
        # Orbit 94 (12-day)
        '2024-10-12', '2024-10-24', '2024-11-05',
        # Orbit 21 - ORPHAN, single acquisition
        '2024-11-15',
    ])
    track_unsorted = np.array(
        [10] * 7 + [65] * 7 + [94] * 3 + [21]
    )
    order = np.argsort(times_unsorted.values)
    times = times_unsorted[order]
    track = track_unsorted[order]
    n_t = len(times)
    n_y, n_x = 8, 10
    rng = np.random.default_rng(0)

    # Sentinel-1 in linear amplitude (will be converted to dB downstream).
    # Values around 0.01-0.5 → about -20 to -3 dB.
    vv_amp = rng.uniform(0.01, 0.4, size=(n_t, n_y, n_x)).astype('float32')
    vh_amp = rng.uniform(0.005, 0.2, size=(n_t, n_y, n_x)).astype('float32')

    # Snowcover varies per timestep — early-season has mixed coverage,
    # late-season fully snow-covered.
    snowcover = np.ones((n_t, n_y, n_x), dtype=int)
    for t in range(min(4, n_t)):
        # patches of bare ground in first few timesteps
        snowcover[t, t % n_y, : (t + 2) % n_x or 1] = 0

    fcf = rng.uniform(0.1, 0.8, size=(n_y, n_x)).astype('float32')
    watermask = np.zeros((n_y, n_x), dtype=int)

    ds = xr.Dataset(
        {
            'vv': (('time', 'y', 'x'), vv_amp),
            'vh': (('time', 'y', 'x'), vh_amp),
            'snowcover': (('time', 'y', 'x'), snowcover),
            'fcf': (('y', 'x'), fcf),
            'watermask': (('y', 'x'), watermask),
        },
        coords={
            'time': times,
            'y': np.linspace(65.7, 65.1, n_y),
            'x': np.linspace(-148.3, -147.4, n_x),
            'track': ('time', track),
        },
        attrs={'s1_units': 'amp'},
    )
    return ds


def test_post_download_pipeline_runs_end_to_end(synthetic_post_download_dataset):
    """Run the entire post-download pipeline against a multi-orbit synthetic
    dataset (including a single-acquisition orbit) and assert plausible
    outputs. This is the offline counterpart to the integration test that
    would catch the snow-index, wet-snow, and preprocessing regressions.
    """
    ds = synthetic_post_download_dataset

    # Preprocessing
    ds = amplitude_to_dB(ds)
    ds = s1_orbit_averaging(ds)
    ds = s1_clip_outliers(ds)

    # Snow-index chain
    ds = calc_delta_cross_ratio(ds, A=2)
    ds = calc_delta_vv(ds)
    ds = calc_delta_gamma(ds, B=0.5)
    ds = clip_delta_gamma_outlier(ds)
    ds = calc_snow_index(ds, ims_masking=True)
    ds = calc_snow_index_to_snow_depth(ds, C=0.55)

    # Wet-snow flags
    ds = id_newly_wet_snow(ds, wet_thresh=-2)
    ds = id_wet_negative_si(ds, wet_SI_thresh=0)
    ds = id_newly_frozen_snow(ds, freeze_thresh=2)
    ds = flag_wet_snow(ds)

    # Sanity checks: outputs exist with expected dims
    for var in ('deltaCR', 'deltavv', 'deltaGamma', 'snow_index',
                'snow_depth', 'wet_snow', 'perma_wet'):
        assert var in ds.data_vars, f"missing {var} after pipeline"
        assert ds[var].sizes['time'] == ds.sizes['time']

    # snow_depth should be non-negative where defined
    sd = ds['snow_depth'].values
    finite_sd = sd[np.isfinite(sd)]
    assert (finite_sd >= 0).all(), "snow_depth must be non-negative"

    # wet_snow should be 0 or 1 (or NaN) — not >1 from the stale-`ts` bug
    wet = ds['wet_snow'].values
    finite_wet = wet[np.isfinite(wet)]
    assert ((finite_wet == 0) | (finite_wet == 1)).all(), (
        "wet_snow must be binary (0/1) wherever defined"
    )
