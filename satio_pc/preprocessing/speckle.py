import numpy as np
from scipy import ndimage
import dask.array as da


def gamma_kernel(img, size, ENL):

    nodata = np.isnan(img)
    img[nodata] = 0

    sig_v2 = 1.0 / ENL
    ENL2 = ENL + 1.
    sfak = 1.0 + sig_v2
    img_mean2 = ndimage.uniform_filter(pow(img, 2), size=size)
    img_mean2[nodata | np.isnan(img_mean2)] = 0.
    img_mean = ndimage.uniform_filter(img, size=size)
    img_mean[nodata | np.isnan(img_mean)] = 0.
    var_z = img_mean2 - pow(img_mean, 2)
    out = img_mean

    with np.errstate(divide='ignore', invalid='ignore'):
        fact1 = var_z / pow(img_mean, 2)
        fact1[np.isnan(fact1)] = 0

        mask = (fact1 > sig_v2) & ((var_z - pow(img_mean, 2) * sig_v2) > 0.)

        if mask.any():
            n = (pow(img_mean, 2) * sfak) / (var_z - pow(img_mean, 2) * sig_v2)
            phalf = (img_mean * (ENL2 - n)) / (2 * n)
            q = ENL * img_mean * img / n
            out[mask] = -phalf[mask] + np.sqrt(pow(phalf[mask], 2) + q[mask])

    out[img == 0 | nodata] = np.nan

    return out


def multitemporal_speckle_filter(stack, kernel, mtwin=7, enl=3):
    """
    stack: np array with multi-temporal stack of backscatter images (linear
    scale)

    kernel: 'mean','gauss','gamma' - 'gamma' is recommended (slower than the
    other kernels though)

    mtwin: filter window size - recommended mtwin=7

    enl: only required for kernel 'gamma' - recommended for S1 enl = 3

    Assumes the data to be in float with nans as nodata.
    """
    nodata = np.isnan(stack)
    stack[nodata] = 0

    layers, rows, cols = stack.shape
    filtim = np.zeros_like(stack, dtype=np.float32)

    rcs = image_sum = image_num = image_fil = None  # pylance unbound warning

    for idx in range(0, layers):
        # Initiate arrays
        if idx == 0:
            image_sum = np.zeros((rows, cols), dtype=np.float32)
            image_num = np.zeros((rows, cols), dtype=np.float32)
            image_fil = np.zeros((layers, rows, cols), dtype=np.float32)

        if kernel == 'mean':
            rcs = ndimage.uniform_filter(
                stack[idx], size=mtwin, mode='mirror')
        elif kernel == 'gauss':
            rcs = ndimage.gaussian_filter(
                stack[idx], mtwin / 4, mode='mirror')
        elif kernel == 'gamma':
            rcs = gamma_kernel(stack[idx], mtwin, enl)

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = (stack[idx] / rcs)
            ratio[np.isnan(ratio)] = 0

        image_sum = image_sum + ratio
        image_num = image_num + (ratio > 0)
        image_fil[idx] = rcs

    with np.errstate(invalid='ignore'):
        for idx in range(0, layers):
            im = stack[idx]
            filtim1 = image_fil[idx] * image_sum / image_num

            filtim1[np.isnan(filtim1)] = 0
            fillmask = (filtim1 == 0) & (im > 0)
            filtim1[fillmask] = im[fillmask]
            mask = im > 0
            filtim1[mask == 0] = im[mask == 0]
            filtim[idx] = filtim1

    filtim[nodata] = np.nan

    return filtim


def _multitemporal_speckle_ts(ts,
                              kernel='gamma',
                              mtwin=7,
                              enl=3):

    nbands = ts.shape[1]

    ts_fil = np.zeros_like(ts)

    for nb in range(nbands):
        stack = ts[:, nb, :, :]
        ts_fil[:, nb, :, :] = multitemporal_speckle_filter(stack,
                                                           kernel,
                                                           mtwin,
                                                           enl)

    return ts_fil


def multitemporal_speckle_ts(dxarr,
                             kernel='gamma',
                             mtwin=7,
                             enl=3):

    chunks = list(dxarr.chunks)

    darr_fil = da.map_blocks(
        _multitemporal_speckle_ts,
        dxarr.data,
        dtype=dxarr.dtype,
        chunks=chunks,
        kernel=kernel,
        mtwin=mtwin,
        enl=enl)

    dxarr_fil = dxarr.copy(data=darr_fil)

    return dxarr_fil
