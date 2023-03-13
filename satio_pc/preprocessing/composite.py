import warnings
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date

import pandas as pd
import numpy as np
import dask.array as da
import xarray as xr


def _get_date_range(start, end, freq, window):

    before = round(window / 2)
    start, end = parse_date(start), parse_date(end)

    date_range = pd.date_range(start=start + timedelta(days=before),
                               end=end,
                               freq=f'{freq}D')
    return date_range


def calculate_moving_composite(darr: xr.DataArray,
                               freq=7,
                               window=None,
                               start=None,
                               end=None,
                               use_all_obs=False):
    """
    Calculate moving median composite of an hyper cube with shape [bands, time,
    rows, cols]

    Parameters
    ----------
    arrs : Numpy 4d array [bands, time, rows, cols]

    start : str or datetime
        start date for the new composites dates
    end : str or datetime
        end date for the new composites dates
    freq : int
        days interval for new dates array
    window : int
        moving window size in days on which the compositing function is applied
    mode : string
        compositing mode. Should be one of 'median', 'mean', 'sum',
        'min' or 'max'
    use_all_obs : bool
        When compositing, the last window might be less than window/2 days
        (or freq/2 if window is None). In this case, some observations might
        get discarded from the compositing function, as the window length
        would be too short. Setting this `True` will include the last
        observations in the last available window, which will then span more
        days than the `window` value. This would avoid discarding observations
        which would be used to increase the SNR of the last window but losing
        temporal resolution.

    Return
    ----------
    Tuple of time_vector and composite 4d array
    """
    window = window or freq  # if window is None, default to `freq` days

    if window < freq:
        raise ValueError('`window` value should be equal or greater than '
                         '`freq` value.')

    before, after = _get_before_after(window)
    date_range = _get_date_range(start, end, freq, before)

    comp_shape = (darr.shape[0], len(date_range),
                  darr.shape[2], darr.shape[3])

    comp = da.zeros(comp_shape,
                    chunks=(1, 1, comp_shape[2], comp_shape[3]),
                    dtype=darr.dtype)
    # comp = da.zeros(comp_shape,
    #                 dtype=darr.dtype)
    time = darr.time.values

    start = str(time[0])[:10] if start is None else start
    end = str(time[-1])[:10] if end is None else end

    date_range = _get_date_range(start, end, freq, before)

    comp_shape = (len(date_range), darr.shape[1],
                  darr.shape[2], darr.shape[3])

    comp = da.zeros(comp_shape,
                    chunks=(1, 1, darr.shape[2], darr.shape[3]),
                    dtype=darr.dtype)

    time = darr.time.values
    intervals_flags = _get_invervals_flags(date_range,
                                           time,
                                           before,
                                           after,
                                           use_all_obs)

    for i, d in enumerate(date_range):
        flag = intervals_flags[i]
        idxs = np.where(flag)[0]

        for band_idx in range(comp.shape[1]):
            comp[i, band_idx, ...] = nanmedian(darr.isel(time=idxs,
                                                         band=band_idx))

    darr_out = xr.DataArray(comp,
                            dims=darr.dims,
                            coords={'time': date_range.values,
                                    'band': darr.band,
                                    'y': darr.y,
                                    'x': darr.x},
                            attrs=darr.attrs)

    return darr_out


def _include_last_obs(idxs):

    # check that all obs are used on last interval
    true_flags = np.where(idxs)[0]
    if true_flags.size:
        last_true_idx = np.where(idxs)[0][-1]
        if last_true_idx != idxs.size - 1:
            idxs[last_true_idx:] = True

    return idxs


def _get_invervals_flags(date_range,
                         time_vector,
                         before,
                         after,
                         use_all_obs):

    intervals_flags = []
    for i, d in enumerate(date_range):
        idxs = interval_flag(
            pd.to_datetime(time_vector),
            d,
            before=before,
            after=after)

        if (i == len(date_range) - 1) and use_all_obs:
            idxs = _include_last_obs(idxs)

        intervals_flags.append(idxs)

    return intervals_flags


def nanmedian(arr):
    """arr should be an xarray with dims (time, y, x)"""

    start_dtype = arr.dtype
    if start_dtype not in (np.float32, np.float64):
        arr = arr.astype(np.float32)
        arr = arr.where(arr != 0, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        res = da.nanmedian(arr.data, axis=0)

    # res comes out as float. nans will be casted to 0s when returning to int
    return res.astype(start_dtype)


def _get_before_after(window: int):
    """
    Returns values for before and after in number of days, given a
    window length.
    """
    half, mod = window // 2, window % 2

    before = after = half

    if mod == 0:  # even window size
        after = max(0, after - 1)  # after >= 0

    return before, after


def interval_flag(time_vector,
                  date: datetime,
                  before: int,
                  after: int):
    """
    Returns a boolean array True where the dates in time_vector fall in an
    interval between data - before and date + after

    Parameters
    ----------
    time_vector : datetime/np.datetime64 array
        time_vector of the source.
    date : datetime/np.datetime64
        target date from which we want to get the neighboring dates from
        time_vector
    """
    midnight = datetime(date.year, date.month, date.day)
    return ((time_vector >= midnight + timedelta(days=-before))
            & (time_vector < midnight + timedelta(days=after + 1)))
