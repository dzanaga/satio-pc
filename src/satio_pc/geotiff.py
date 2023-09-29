import re
import os
from pathlib import Path
from typing import List, Dict

from loguru import logger
import numpy as np
import rasterio
from rasterio.profiles import Profile
from rasterio.crs import CRS
import xarray as xr

# from satio.utils.geotiff import (DefaultProfile,
#                                  write_geotiff)
# from satio.utils.geotiff import get_rasterio_profile_shape

VALUE_LIMITS = {
    'uint8': {
        'min_value': 0,
        'max_value': 253,
        'nodata_value': 255
    },
    'uint16': {
        'min_value': 0,
        'max_value': 65533,
        'nodata_value': 65535
    },
    'uint13': {
        'min_value': 0,
        'max_value': 8189,
        'nodata_value': 8191
    },
    'uint14': {
        'min_value': 0,
        'max_value': 16381,
        'nodata_value': 16383
    }
}


def slash_tile(tile: str):

    if len(tile) != 5:
        raise ValueError(f"tile should be a str of len 5, not {tile}")

    return f"{tile[:2]}/{tile[2]}/{tile[3:]}"


class DefaultProfile(Profile):
    """Tiled, band-interleaved, LZW-compressed, 8-bit GTiff."""

    defaults = {
        'driver': 'GTiff',
        'interleave': 'band',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
        'compress': 'deflate',
        'dtype': 'float32'
    }


def get_rasterio_profile_shape(shape,
                               bounds,
                               epsg,
                               dtype,
                               blockxsize=1024,
                               blockysize=1024,
                               **params):

    base_profile = DefaultProfile()

    if len(shape) == 2:
        shape = [1] + shape

    count, height, width = shape

    crs = CRS.from_epsg(epsg)

    base_profile.update(
        transform=rasterio.transform.from_bounds(*bounds,
                                                 width=width,
                                                 height=height),
        width=width,
        height=height,
        blockxsize=blockxsize,
        blockysize=blockysize,
        dtype=dtype,
        crs=crs,
        count=count)

    base_profile.update(**params)

    return base_profile


def compress_data(arr, dtype, *, min_value, max_value, nodata_value):

    offsets = np.nanmin(arr, axis=(1, 2)) - min_value
    offsets = np.expand_dims(offsets, (1, 2))
    arr2 = arr - np.broadcast_to(offsets, arr.shape)

    scales = np.nanmax(arr2, axis=(1, 2)) / max_value
    scales = np.expand_dims(scales, (1, 2))
    with np.errstate(divide='ignore', invalid='ignore'):
        arr2 = arr2 / np.broadcast_to(scales, arr.shape)

    arr2[~np.isfinite(arr)] = nodata_value

    return arr2.round().astype(dtype), np.squeeze(scales), np.squeeze(offsets)


def restore_data(arr, scales, offsets, nodata_value):
    """
    scales == max_vals
    offsets == min_vals
    """
    arr = arr.astype(np.float32)
    arr[arr == nodata_value] = np.nan

    scales = np.expand_dims(scales, (1, 2))
    arr = arr * np.broadcast_to(scales, arr.shape)

    offsets = np.expand_dims(offsets, (1, 2))
    arr = arr + np.broadcast_to(offsets, arr.shape)

    return arr


def write_geotiff_tags(arr,
                       profile,
                       filename,
                       bands_names=None,
                       colormap=None,
                       nodata=None,
                       tags=None,
                       bands_tags=None,
                       scales=None,
                       offsets=None):
    """
    tags should be a dictionary
    bands_tags should be a list of dictionarites with len == arr.shape[0]
    """
    bands_tags = bands_tags if bands_tags is not None else []

    if nodata is not None:
        profile.update(nodata=nodata)

    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=0)

    if os.path.isfile(filename):
        os.remove(filename)

    with rasterio.open(filename, 'w', **profile) as dst:
        dst.write(arr)

        if tags is not None:
            dst.update_tags(**tags)

        for i, bt in enumerate(bands_tags):
            dst.update_tags(i + 1, **bt)

        if colormap is not None:
            dst.write_colormap(
                1, colormap)

        if scales is not None:
            dst.scales = scales

        if offsets is not None:
            dst.offsets = offsets

        if bands_names is not None:
            if len(bands_names) != arr.shape[0]:
                logger.warning("`bands_names` should be the same length as "
                               "the number of bands in the array. Cannot "
                               "set band descriptions.")
            else:
                for i, b in enumerate(bands_names):
                    dst.set_band_description(i + 1, b)


def save_features_geotiff(data,
                          bounds: List = [0, 1, 0, 1],
                          epsg: int = 4326,
                          bands_names: List = None,
                          filename: str = None,
                          tags: Dict = None,
                          compress_tag: str = 'deflate-uint16',
                          **profile_kwargs):
    """Save geotiff of 3d features array. Sets the band names (first dimension)
    as bands description.

    Args:
        data (np.ndarray): numpy array with dims (band, y, x)
        bounds (List, optional): _description_. Defaults to [0, 1, 0, 1].
        epsg (int, optional): _description_. Defaults to 4326.
        filename (str, optional): _description_. Defaults to None.
        tags (Dict, optional): _description_. Defaults to None.
        compress_tag (str, optional): _description_. Defaults to 'deflate-uint16'.

    """
    compress_profile, dtype_value_limits, dtype = get_compression_profile(
        compress_tag)

    logger.info(f"Saving {filename}...")

    data, scales, offsets = compress_data(data, dtype, **dtype_value_limits)

    profile = get_rasterio_profile_shape(data.shape, bounds,
                                         epsg, dtype)

    profile.update(nodata=dtype_value_limits['nodata_value'])
    profile.update(**compress_profile)
    profile.update(**profile_kwargs)

    scales = np.squeeze(scales).tolist()
    offsets = np.squeeze(offsets).tolist()

    default_tags = {
        'bands': bands_names,
    }

    tags = tags or {}
    tags = {**default_tags, **tags}

    if filename is not None:
        write_geotiff_tags(data, profile, filename,
                           bands_names=bands_names,
                           tags=tags,
                           scales=scales, offsets=offsets)

    return data, scales, offsets


def load_features_geotiff(feat_fn):

    with rasterio.open(feat_fn) as src:
        arr = src.read()
        scales = src.scales
        offsets = src.offsets
        nodata = src.nodata

        bands = eval(src.tags()['bands'])

        arr = restore_data(arr, scales, offsets, nodata)
        bounds = src.bounds
        epsg = src.crs.to_epsg()

    attrs = {'epsg': epsg,
             'bounds': bounds}

    new_y, new_x = compute_pixel_coordinates(bounds, arr.shape[-2:])

    darr = xr.DataArray(arr,
                        dims=['band', 'y', 'x'],
                        coords={'band': bands,
                                'y': new_y,
                                'x': new_x},
                        attrs=attrs)

    return darr


def compute_pixel_coordinates(bounds, shape):
    """
    Compute the y and x coordinates for every pixel in an image.

    Args:
    bounds (tuple): A tuple containing (xmin, ymin, xmax, ymax).
    shape (tuple): A tuple containing the image shape (rows, columns).

    Returns:
    tuple: Two arrays containing y and x coordinates for every pixel.
    """
    xmin, ymin, xmax, ymax = bounds
    rows, cols = shape

    x_res = (xmax - xmin) / cols
    y_res = (ymax - ymin) / rows

    if x_res != y_res:
        raise ValueError("Different resolution for y and x axis are not "
                         "supported. Bounds and shape are not consistent "
                         "with the same resolution on both axis.")

    res_half = x_res / 2

    xx = np.linspace(xmin + res_half,
                     xmax - res_half,
                     cols)

    yy = np.linspace(ymax - res_half,
                     ymin + res_half,
                     rows)

    return yy, xx


def _get_jp2_compression_profile(compress_tag):
    """ compress_tag e.g. 'jp2-uint{nbits}-q{quality}'
    tag = f'uint{nbits}-deflate-z{z}-lsb{lsb}'
    dtype_tag = f'uint{nbits}'
    """

    mbits = re.search(r'-uint(\d*)', compress_tag)
    nbits = int(mbits.group(1)) if mbits else 16

    mqual = re.search(r'-q(\d*)', compress_tag)
    quality = int(mqual.group(1)) if mqual else 100

    dtype = np.uint16 if nbits > 8 else np.uint8

    profile = {'driver': 'JP2OpenJPEG',
               'USE_TILE_AS_BLOCK': True,
               'quality': quality,
               'reversible': False,
               'resolutions': 1,
               'nbits': nbits}

    dtype_tag = f'uint{nbits}'

    if dtype_tag not in VALUE_LIMITS.keys():
        raise ValueError(f"dtype tag {dtype_tag} not supported. "
                         f"Available profiles: {list(VALUE_LIMITS.keys())}")

    value_limits = VALUE_LIMITS[dtype_tag]

    return profile, value_limits, dtype


def _get_deflate_compression_profile(compress_tag):
    """ compress_tag e.g. 'jp2-uint{nbits}-q{quality}'
    """

    mbits = re.search(r'-uint(\d*)', compress_tag)
    nbits = int(mbits.group(1)) if mbits else 16

    mlsb = re.search(r'-lsb(\d*)', compress_tag)
    lsb = int(mlsb.group(1)) if mlsb else None

    mz = re.search(r'-z(\d*)', compress_tag)
    z = int(mz.group(1)) if mz else 6

    dtype = np.uint16 if nbits > 8 else np.uint8

    profile = {'tiled': False,
               'compress': 'deflate',
               'interleave': 'band',
               'predictor': 2,
               'discard_lsb': lsb,
               'zlevel': z,
               'nbits': nbits}

    dtype_tag = f'uint{nbits}'

    if dtype_tag not in VALUE_LIMITS.keys():
        raise ValueError(f"dtype tag {dtype_tag} not supported. "
                         f"Available profiles: {list(VALUE_LIMITS.keys())}")

    value_limits = VALUE_LIMITS[dtype_tag]
    return profile, value_limits, dtype


def get_compression_profile(compress_tag):
    """ compress_tag e.g. 'jp2-uint{nbits}-q{quality}'
        or 'deflate-uint{nbits}-lsb{lsb}-z{zvalue}
    """
    if compress_tag.startswith('jp2'):
        return _get_jp2_compression_profile(compress_tag)
    elif compress_tag.startswith('deflate'):
        return _get_deflate_compression_profile(compress_tag)
    else:
        raise ValueError("Compress tag not recognized")
