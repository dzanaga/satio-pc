import warnings
import numpy as np
import skimage
import dask.array as da
import xarray as xr


def _rescale_ts(ts,
                scale=2,
                order=1,
                preserve_range=True,
                nodata_value=None,
                sigma=0.5):

    if order > 1:
        raise ValueError('Skimage giving issues with nans and cubic interp')

    ts_dtype = ts.dtype
    is_float = ts_dtype in [np.float32, np.float64]

    if (order > 0) and not is_float:
        # if data is not float we take care of the nodata
        new_dtype = np.float32
        ts = ts.astype(new_dtype)
        if nodata_value is not None:
            ts[ts == nodata_value] = np.nan
    else:
        new_dtype = ts_dtype

    shape = ts.shape
    new_shape = shape[0], shape[1], int(
        shape[2] * scale), int(shape[3] * scale)
    new = np.empty(new_shape, dtype=new_dtype)

    anti_aliasing = None
    anti_aliasing_sigma = None

    if scale < 1:
        anti_aliasing = True,
        anti_aliasing_sigma = sigma

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in range(shape[0]):
            new[t, :, :, :] = skimage.transform.rescale(
                ts[t, ...],
                scale=scale,
                order=order,
                preserve_range=preserve_range,
                channel_axis=0,
                anti_aliasing=anti_aliasing,
                anti_aliasing_sigma=anti_aliasing_sigma)

    if nodata_value is not None:
        new[da.isnan(new)] = nodata_value

    new = new.astype(ts_dtype)

    return new


def rescale_ts(ds20,
               scale=2,
               order=1,
               preserve_range=True,
               nodata_value=0,
               sigma=0.5):

    if isinstance(ds20.data, da.core.Array):
        chunks = list(ds20.chunks)

        for i in -1, -2:
            chunks[i] = tuple(map(lambda x: x * scale, chunks[i]))

        darr_scaled = da.map_blocks(
            _rescale_ts,
            ds20.data,
            dtype=ds20.dtype,
            chunks=chunks,
            scale=scale,
            order=order,
            preserve_range=preserve_range,
            nodata_value=nodata_value,
            sigma=sigma)

    else:
        darr_scaled = _rescale_ts(ds20.data,
                                  scale=scale,
                                  order=order,
                                  preserve_range=preserve_range,
                                  nodata_value=nodata_value,
                                  sigma=sigma)

    xmin, ymin, xmax, ymax = ds20.ewc.bounds
    new_res = ds20.attrs.get('resolution',
                             ds20.x[1] - ds20.x[0]) / scale
    new_res_half = new_res / 2

    new_x = np.linspace(xmin + new_res_half,
                        xmax - new_res_half,
                        int(ds20.shape[-2] * scale))

    new_y = np.linspace(ymax - new_res_half,
                        ymin + new_res_half,
                        int(ds20.shape[-1] * scale))

    ds20u = xr.DataArray(darr_scaled,
                         dims=ds20.dims,
                         coords={'time': ds20.time,
                                 'band': ds20.band,
                                 'y': new_y,
                                 'x': new_x},
                         attrs=ds20.attrs)

    ds20u.attrs['resolution'] = new_res

    return ds20u
