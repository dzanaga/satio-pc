import numba
import numpy as np

# def cubic_spline_interpolation(x, y, xx):
#     # Compute the second derivatives of the cubic spline
#     n = len(x)
#     h = np.zeros(n - 1)
#     b = np.zeros(n)
#     u = np.zeros(n - 1)
#     v = np.zeros(n - 1)
#     z = np.zeros(n)
#     for i in range(n - 1):
#         h[i] = x[i + 1] - x[i]
#         b[i] = (y[i + 1] - y[i]) / h[i]
#     for i in range(1, n - 1):
#         u[i] = h[i - 1] / (h[i - 1] + h[i])
#         v[i] = h[i] / (h[i - 1] + h[i])
#         z[i] = 3 * (u[i] * b[i - 1] + v[i] * b[i])
#     z[0] = z[n - 1] = 0
#     for i in range(1, n):
#         k = n - i - 1
#         z[k] = (z[k] - u[k] * z[k + 1]) / (1 - u[k] * v[k])
#     # Interpolate the function at the given points
#     m = len(xx)
#     yy = np.zeros(m)
#     for i in range(m):
#         j = np.searchsorted(x, xx[i], side='right') - 1
#         if j < 0 or j >= n - 1:
#             yy[i] = np.nan
#         else:
#             t = (xx[i] - x[j]) / h[j]
#             y1 = y[j]
#             y2 = y[j + 1]
#             z1 = z[j]
#             z2 = z[j + 1]
#             a = z1 * h[j] - (y2 - y1)
#             b = -z2 * h[j] + (y2 - y1)
#             q = (1 - t) * y1 + t * y2 + t * (1 - t) * (a * (1 - t) + b * t)
#             yy[i] = q
#     return yy

# @nb.jit(nopython=True)
# def linear_interpolation(x, y, xx):
#     m = len(xx)
#     yy = np.zeros(m)
#     for i in range(m):
#         j = np.searchsorted(x, xx[i], side='right') - 1
#         if j < 0 or j >= len(x) - 1:
#             yy[i] = np.nan
#         else:
#             t = (xx[i] - x[j]) / (x[j + 1] - x[j])
#             yy[i] = (1 - t) * y[j] + t * y[j + 1]
#     return yy


@numba.jit(nopython=True, error_model='numpy', fastmath=True)
def interpolate_ts_linear(arr):
    out = np.zeros_like(arr)
    for band in range(arr.shape[1]):
        for py in range(arr.shape[2]):
            for px in range(arr.shape[3]):

                t = arr[:, band, py, px].copy()

                nans_ids = (t == 0)
                x_valid = np.where(~nans_ids)[0]
                y_valid = t[x_valid]
                x_invalid = np.where(nans_ids)[0]

                re_init = False
                if t[0] == 0:
                    t[0] = np.mean(y_valid)
                    re_init = True

                if t[-1] == 0:
                    t[-1] = np.mean(y_valid)
                    re_init = True

                if re_init:
                    nans_ids = (t == 0)
                    x_valid = np.where(~nans_ids)[0]
                    y_valid = t[x_valid]
                    x_invalid = np.where(nans_ids)[0]

                y_new = t
                m = len(x_invalid)
                for i in range(m):
                    j = np.searchsorted(
                        x_valid, x_invalid[i], side='right') - 1
                    
                    q = (x_valid[j + 1] - x_valid[j])
                    s = (x_invalid[i] - x_valid[j]) / q
                    
                    y_new[x_invalid[i]] = (1 - s) * y_valid[j] + s * y_valid[j + 1]

                out[:, band, py, px] = y_new

    return out
