import warnings
from typing import List
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date

from loguru import logger
import pandas as pd
import numpy as np
import dask.array as da


def _nanminmax(arr, axis, f=np.nanmin):
    """
    Avoid np.nanmin/np.nanmax failing when computing an empty slice
    """
    if arr.size == 0:
        return np.zeros(arr.shape[1:], dtype=np.float32) * np.nan
    else:
        return f(arr, axis)


def nanmin(arr, axis=None):
    return _nanminmax(arr, axis, np.nanmin)


def nanmax(arr, axis=None):
    return _nanminmax(arr, axis, np.nanmax)


COMPUTE_META = {'median': da.nanmedian,
                'mean': np.nanmean,
                'sum': np.nansum,
                'min': nanmin,
                'max': nanmax}

SUPPORTED_MODES = list(COMPUTE_META.keys())


def interval_flag(time_vector: List[datetime],
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


def _get_date_range(start, end, freq, before):

    start, end = parse_date(start), parse_date(end)

    date_range = pd.date_range(start=start + timedelta(days=before),
                               end=end,
                               freq=f'{freq}D')
    return date_range


def calculate_moving_composite(arrs: np.array,
                               time_vector,
                               start='20171001',
                               end='20190401',
                               mode='median',
                               freq=7,
                               window=None,
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

    window = window or freq  # is window is None, default to `freq` days

    if window < freq:
        raise ValueError('`window` value should be equal or greater than '
                         '`freq` value.')

    before, after = _get_before_after(window)
    date_range = _get_date_range(start, end, freq, before)

    original_ndim = arrs.ndim

    if arrs.ndim == 3:
        arrs = np.expand_dims(arrs, 0)

    comp_shape = (arrs.shape[0], len(date_range),
                  arrs.shape[2], arrs.shape[3])

    comp = np.zeros(comp_shape, dtype=arrs.dtype)

    intervals_flags = _get_invervals_flags(date_range,
                                           time_vector,
                                           before,
                                           after,
                                           use_all_obs)
    for i, d in enumerate(date_range):
        idxs = intervals_flags[i]

        for band_idx in range(comp.shape[0]):

            comp[band_idx, i, ...] = _compositing_f(arrs[band_idx, idxs, ...],
                                                    mode=mode)

    date_range = date_range.to_pydatetime()

    if original_ndim == 3:
        comp = comp[0]

    return comp, date_range  # type: ignore


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


def _compositing_f(arr, mode='median'):

    compute_f = COMPUTE_META.get(mode)

    if compute_f is None:
        raise ValueError(f'`mode` should be one of {SUPPORTED_MODES}.'
                         f'Not `{mode}`')

    start_dtype = arr.dtype
    if start_dtype not in (np.float32, np.float64):
        arr = arr.astype(np.float32)
        arr[arr == 0] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        res = compute_f(arr, axis=0)

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


def _check_window_settings(freq,
                           before,
                           after,
                           mode,
                           supported_modes=None):
    """
    Perform a check on the values of before and after against freq, mode
    and supported_modes.

    If mode is sum before and after will be computed to give non overlapping
    windows.

    If one of before or after are None, the other is set to the values of the
    valid one.

    If before and after are both None,
    """
    if supported_modes is None:
        supported_modes = SUPPORTED_MODES

    if mode not in supported_modes:
        raise ValueError(('Compositing mode should be one of '
                          f'{supported_modes}, but got: `{mode}`'))

    no_ov_modes = ['sum', 'min', 'max']
    if mode in no_ov_modes:
        if (before is not None or after is not None):
            logger.warning(('`before` and `after` arguments are ignored for '
                            f'compositing mode `{no_ov_modes}`.'))
        # For these modes window overlap is not allowed in the time subsets
        # to avoid double counting of values
        before = after = None

    # if one of before or after is None, set it simmetrically to the valid one
    before = before or after
    after = after or before

    if (before is None and after is None):
        before, after = _get_default_before_after(freq)

    return before, after


def _get_default_before_after(freq):
    """
    Based on the freq, return before and after for non overlapping windows
    """
    if freq == 1:
        before = 0
        after = 0
    elif freq % 2 == 0:
        before = freq / 2 - 1
        after = freq / 2
    else:
        before = int(np.floor(freq / 2))
        after = int(np.floor(freq / 2))

    return before, after

# old implementation - to be removed at some point...


def _calculate_percentile(arrs, arr_dim, percent):

    if arrs.dtype == np.uint16:
        from satio.utils.nonzeropercentile import (hyper_nonzeropercentile,
                                                   nonzeropercentile)

        hyper_n_percentile = hyper_nonzeropercentile
        n_percentile = nonzeropercentile

    elif arrs.dtype == np.float32:
        from satio.utils.nanpercentile import (hyper_nanpercentile,
                                               nanpercentile)

        hyper_n_percentile = hyper_nanpercentile
        n_percentile = nanpercentile
    else:
        raise TypeError(f'{arrs.dtype} not supported for numba nan/nonzero '
                        'percentiles calculation. Should be uint16 or float32')

    if arr_dim == 4:
        return hyper_n_percentile(arrs, percent)
    elif arr_dim == 3:
        return n_percentile(arrs, percent)


def _calculate_mean(arrs, arr_dim):
    input_dtype = arrs.dtype
    mean = None  # turn off pylance warning

    if input_dtype != np.float32 or input_dtype != np.float64:
        arrs = arrs.astype(np.float32)
        arrs[arrs == 0] = np.nan
    if arr_dim == 4:
        mean = np.nanmean(arrs, axis=1)
    elif arr_dim == 3:
        mean = np.nanmean(arrs, axis=0)
    if input_dtype != np.float32 or input_dtype != np.float64:
        mean[np.isnan(mean)] = 0
    mean = mean.astype(input_dtype)

    return mean


def _calculate_sum(arrs, arr_dim):
    input_dtype = arrs.dtype
    mean = None
    if input_dtype != np.float32 or input_dtype != np.float64:
        arrs = arrs.astype(np.float32)
        arrs[arrs == 0] = np.nan
    if arr_dim == 4:
        mean = np.nansum(arrs, axis=1)
    elif arr_dim == 3:
        mean = np.nansum(arrs, axis=0)
    if input_dtype != np.float32 or input_dtype != np.float64:
        mean[np.isnan(mean)] = 0
    mean = mean.astype(input_dtype)

    return mean


def calculate_moving_composite_old(arrs,
                                   time_vector,
                                   mode='percentile',
                                   start='20171001',
                                   end='20190401',
                                   freq=7,
                                   before=7,
                                   after=7,
                                   percent=50):
    """
    Calculate composite of an hyper cube with shape [bands, time,
    rows, cols]. Supported modes are: moving percentile, moving mean,
    and a temporal sum

    Parameters
    ----------
    arrs : Numpy 4d array [bands, time, rows, cols]
    time_vector : datetime array specifying the time dimension of arrs
    mode : one or [percentile, mean, sum]

    start : str or datetime
        start date for the new composites dates
    end : str or datetime
        end date for the new composites dates
    freq : int
        days interval for new dates array
    before: int
        window before date for the compositing
    after: int
        window after date for the compositing
    percent: int
        between 0 and 100. Usually 50 for the median compositing

    Return
    ----------
    Tuple of time_vector and composite 4d array
    """

    supported_modes = ['percentile', 'mean', 'sum']

    if mode not in supported_modes:
        raise ValueError(('Compositing mode should be one of '
                          f'[`percentile`, `mean`, `sum`], but got: `{mode}`'))

    if mode == 'sum':
        if (before is not None or after is not None):
            logger.warning(('`before` and `after` arguments are ignored for '
                            f'compositing mode `sum`.'))
        # Sum is special, as we don't allow overlap in the time subsets
        # to avoid double counting of values
        if freq == 1:
            before = 0
            after = 0
        elif freq % 2 == 0:
            before = freq / 2 - 1
            after = freq / 2
        else:
            before = int(np.floor(freq / 2))
            after = int(np.floor(freq / 2))

    # Setup requested composite date range
    date_range = pd.date_range(start=start, end=end, freq='{}D'.format(freq))

    # Calculate composite values
    arr_dim = len(arrs.shape)
    if arr_dim == 4:
        comp = np.zeros((arrs.shape[0], len(date_range),
                         arrs.shape[2], arrs.shape[3]), dtype=arrs.dtype)

        for i, d in enumerate(date_range):
            idxs = interval_flag(pd.to_datetime(
                time_vector), d, before=before, after=after)

            if mode == 'percentile':
                comp[:, i, ...] = _calculate_percentile(arrs[:, idxs, ...],
                                                        arr_dim,
                                                        percent)
            elif mode == 'mean':
                comp[:, i, ...] = _calculate_mean(arrs[:, idxs, ...],
                                                  arr_dim)
            elif mode == 'sum':
                comp[:, i, ...] = _calculate_sum(arrs[:, idxs, ...],
                                                 arr_dim)

    elif arr_dim == 3:
        comp = np.zeros((len(date_range),
                         arrs.shape[1], arrs.shape[2]), dtype=arrs.dtype)

        for i, d in enumerate(date_range):
            idxs = interval_flag(pd.to_datetime(
                time_vector), d, before=before, after=after)

            if mode == 'percentile':
                comp[i, ...] = _calculate_percentile(arrs[idxs, ...],
                                                     arr_dim,
                                                     percent)
            elif mode == 'mean':
                comp[i, ...] = _calculate_mean(arrs[idxs, ...],
                                               arr_dim)
            elif mode == 'sum':
                comp[i, ...] = _calculate_sum(arrs[idxs, ...],
                                              arr_dim)

    date_range = date_range.to_pydatetime()

    return comp, date_range  # type: ignore
