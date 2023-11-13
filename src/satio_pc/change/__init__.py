import numpy as np
from numba import njit


@njit
def pearsonr(x, y):
    mx = x.mean()
    my = y.mean()
    num = np.sum((x - mx) * (y - my))
    den = np.sqrt(np.sum((x - mx) ** 2) * np.sum((y - my) ** 2))
    return num / den


@njit
def pearsonr_rolling(imx, imy, kernel_size=7):
    w = (kernel_size - 1) // 2
    nbands, nrows, ncols = imx.shape
    out = np.zeros((nrows, ncols), dtype=np.float32)
    for i in range(nrows):
        row_start = max(0, i - w)
        row_end = min(nrows - 1, i + w)
        for j in range(ncols):
            col_start = max(0, j - w)
            col_end = min(ncols - 1, j + w)
            x = imx[:, row_start:row_end, col_start:col_end]
            y = imy[:, row_start:row_end, col_start:col_end]
            out[i, j] = pearsonr(x, y)
    return out


@njit
def pearsonr_bands(x, y):
    nbands = x.shape[0]
    band_correlation = np.zeros(nbands,
                                dtype=np.float32)
    for b in range(nbands):
        band_correlation[b] = pearsonr(x[b], y[b])
    return band_correlation


@njit
def pearsonr_rolling_band(imx, imy, kernel_size=7, method='mean'):
    w = (kernel_size - 1) // 2
    nbands, nrows, ncols = imx.shape
    out = np.zeros((nrows, ncols), dtype=np.float32)
    for i in range(nrows):
        row_start = max(0, i - w)
        row_end = min(nrows - 1, i + w)
        for j in range(ncols):
            col_start = max(0, j - w)
            col_end = min(ncols - 1, j + w)
            x = imx[:, row_start:row_end, col_start:col_end]
            y = imy[:, row_start:row_end, col_start:col_end]
            if method == 'mean':
                out[i, j] = pearsonr_bands(x, y).mean()
            elif method == 'min':
                out[i, j] = pearsonr_bands(x, y).min()
    return out
