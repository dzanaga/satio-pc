import warnings
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date

import pandas as pd
import numpy as np
import dask.array as da
import xarray as xr
from loguru import logger


SUPPORTED_MODES = ['median', 'mean', 'sum', 'min', 'max']


def nonzero_reducer(arr, mode):
    """Compute nanmedian on array. If array is not float32/float64,
    the array is converted to float and 0 values are skipped.

    mode can be 'median', 'mean', 'sum' """

    if isinstance(arr, da.core.Array):
        reducer_lib = da
    else:
        reducer_lib = np

    reducer_func = {'median': reducer_lib.nanmedian,
                    'mean': reducer_lib.nanmean,
                    'sum': reducer_lib.nansum,
                    'min': reducer_lib.nanmin,
                    'max': reducer_lib.nanmax}.get(mode)

    if reducer_func is None:
        raise ValueError(f"Compositing mode '{mode}' not supported. "
                         f"Should be one of {SUPPORTED_MODES}.")

    start_dtype = arr.dtype
    if start_dtype not in (np.float32, np.float64):
        arr = arr.astype(np.float32)
        arr = arr.where(arr != 0, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        res = reducer_func(arr.data, axis=0)

    # res comes out as float. nans will be casted to 0s when returning to int
    return res.astype(start_dtype)


def _get_date_range(start, end, freq, window):

    before, after = _get_before_after(window)

    start, end = parse_date(start), parse_date(end)

    date_range = pd.date_range(start=start + timedelta(days=before),
                               end=end + timedelta(days=-after),
                               freq=f'{freq}D')
    return date_range


def calculate_moving_composite(darr: xr.DataArray,
                               freq=7,
                               window=None,
                               start=None,
                               end=None,
                               mode='median',
                               use_all_obs=True,
                               ):
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
        which would be used to increase the SNR of the last window.

    Return
    ----------
    Tuple of time_vector and composite 4d array
    """
    if mode not in SUPPORTED_MODES:
        raise ValueError(('Compositing mode should be one of '
                          f'{SUPPORTED_MODES}, but got: `{mode}`'))

    no_ov_modes = ['sum', 'min', 'max']
    if mode in no_ov_modes:
        if window is not None and window != freq:
            logger.warning(('`window` argument is ignored for '
                            f'compositing mode `{no_ov_modes}`.'))
        # For these modes window overlap is not allowed in the time subsets
        # to avoid double counting of values
        window = freq
    window = window or freq  # if window is None, default to `freq` days

    if window < freq:
        raise ValueError('`window` value should be equal or greater than '
                         '`freq` value.')

    time = darr.time.values

    start = str(time[0])[:10] if start is None else start
    end = str(time[-1])[:10] if end is None else end

    date_range = _get_date_range(start, end, freq, window)

    comp_shape = (len(date_range), darr.shape[1],
                  darr.shape[2], darr.shape[3])

    if isinstance(darr.data, da.core.Array):
        comp = da.zeros(comp_shape,
                        chunks=(1, 1, darr.shape[2], darr.shape[3]),
                        dtype=darr.dtype)
    else:
        comp = np.zeros(comp_shape,
                        dtype=darr.dtype)

    time = darr.time.values
    before, after = _get_before_after(window)
    intervals_flags = _get_invervals_flags(date_range,
                                           time,
                                           before,
                                           after,
                                           use_all_obs)

    for i, d in enumerate(date_range):
        flag = intervals_flags[i]
        if not any(flag):
            continue
        idxs = np.where(flag)[0]
        comp[i, ...] = nonzero_reducer(darr.isel(time=idxs), mode)

    darr_out = xr.DataArray(comp,
                            dims=darr.dims,
                            coords={'time': date_range.values,
                                    'band': darr.band,
                                    'y': darr.y,
                                    'x': darr.x},
                            attrs=darr.attrs)

    return darr_out


def _get_invervals_flags(date_range,
                         time_vector,
                         before,
                         after,
                         use_all_obs):

    intervals_flags = []
    for i, d in enumerate(date_range):

        if i == len(date_range) - 1 and use_all_obs:
            tmp_after = None  # defaults to end of timeseries
        else:
            tmp_after = after

        idxs = interval_flag(
            pd.to_datetime(time_vector),
            d,
            before=before,
            after=tmp_after)

        intervals_flags.append(idxs)

    return intervals_flags


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

    if after is None:
        return time_vector >= midnight + timedelta(days=-before)
    else:
        return ((time_vector >= midnight + timedelta(days=-before))
                & (time_vector < midnight + timedelta(days=after + 1)))
