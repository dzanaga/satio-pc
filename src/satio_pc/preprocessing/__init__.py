import numpy as np
import xarray as xr


def to_dataarray(arr,
                 bounds=None,
                 epsg=None,
                 bands=None,
                 time=None,
                 attrs=None):

    if bounds is None:
        bounds = [0, 0, 1, 1]
    if bands is None and arr.ndim == 3:
        bands = [str(i) for i in range(arr.shape[0])]
    if time is None and arr.ndim == 4:
        time = [i for i in range(arr.shape[0])]

    xmin, ymin, xmax, ymax = bounds
    resolution_x = (xmax - xmin) / arr.shape[-1]
    resolution_y = (ymax - ymin) / arr.shape[-2]
    y = np.linspace(ymax - resolution_y / 2,
                    ymin + resolution_y / 2,
                    arr.shape[-2])
    x = np.linspace(xmin + resolution_x / 2,
                    xmax - resolution_x / 2,
                    arr.shape[-1])

    if arr.ndim == 4:
        darr = xr.DataArray(arr,
                            dims=['time', 'band', 'y', 'x'],
                            coords={'time': time,
                                    'y': y,
                                    'x': x,
                                    'band': bands})
    elif arr.ndim == 3:
        darr = xr.DataArray(arr,
                            dims=['band', 'y', 'x'],
                            coords={'y': y,
                                    'x': x,
                                    'band': bands})
    elif arr.ndim == 2:
        darr = xr.DataArray(arr,
                            dims=['y', 'x'],
                            coords={'y': y,
                                    'x': x})

    else:
        raise ValueError("Array must be 2D or 3D or 4D")
    new_attrs = dict(bounds=bounds)
    if epsg is not None:
        new_attrs['epsg'] = epsg

    if attrs is None:
        attrs = new_attrs
    else:
        attrs.update(new_attrs)

    darr.attrs = attrs

    return darr


def get_yx_coords(arr, bounds):
    xmin, ymin, xmax, ymax = bounds
    resolution_x = (xmax - xmin) / arr.shape[-1]
    resolution_y = (ymax - ymin) / arr.shape[-2]
    y = np.linspace(ymax - resolution_y / 2,
                    ymin + resolution_y / 2,
                    arr.shape[-2])
    x = np.linspace(xmin + resolution_x / 2,
                    xmax - resolution_x / 2,
                    arr.shape[-1])
    return y, x


def to_dataarray_coords(arr,
                        dims,
                        coords,
                        attrs=None):

    darr = xr.DataArray(arr,
                        dims=dims,
                        coords=coords.coords)

    if attrs is not None:
        darr.attrs = attrs

    return darr
