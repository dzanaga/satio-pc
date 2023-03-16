"""
SCL mask improvements

by checking the values temporally, we can identify anomalies

for example if only few pixels have a high rate of being flagged as
either snow/dark/cloud and the block cover for that category is low,
then we can assume that this is an outlier.

For example, one river pixel is marked as dark/shadow 90% of the times
for those observations where the shadow pixels are a small amount that means
that probably there are no clouds/shadows and the flag can be removed

We can run some outliers detection based on this.

Snow fraction. We should compute the snow fraction over the valid obs (non
clouds)
This can be used for permanent snow detection, and in general can be an
interesting layer.
"""
from dataclasses import dataclass

import dask
import dask.array as da
from dask_image.ndmorph import binary_erosion, binary_dilation
from skimage.morphology import footprints

SCL_LEGEND = {
    'no_data': 0,
    'saturated_or_defective': 1,
    'dark_area_pixels': 2,
    'cloud_shadows': 3,
    'vegetation': 4,
    'not_vegetated': 5,
    'water': 6,
    'unclassified': 7,
    'cloud_medium_probability': 8,
    'cloud_high_probability': 9,
    'thin_cirrus': 10,
    'snow': 11
}

SCL_MASK_VALUES = [1, 3, 8, 9, 10, 11]


@dataclass
class SCLMask:
    """
    Container for processed SCL mask timeseries DataArray.

    Attributes:
    mask
    obs
    invalid_before
    invalid_after

    Default SCL_MASK_VALUES = [1, 3, 8, 9, 10, 11]
    """
    mask: dask.array.core.Array
    obs: dask.array.core.Array
    invalid_before: dask.array.core.Array
    invalid_after: dask.array.core.Array
    cover_snow: dask.array.core.Array
    cover_water: dask.array.core.Array
    cover_veg: dask.array.core.Array
    cover_notveg: dask.array.core.Array

    def __repr__(self):
        return f'<SCLMask container - mask.shape: {self.mask.shape}>'


def scl_to_mask(scl_data,
                mask_values=None,
                erode_r=None,
                dilate_r=None,
                max_invalid_ratio=None):
    """
    From a timeseries (t, y, x) dataarray returns a binary mask False for the
    given mask_values and True elsewhere (valid pixels).

    SCL_LEGEND = {
        'no_data': 0,
        'saturated_or_defective': 1,
        'dark_area_pixels': 2,
        'cloud_shadows': 3,
        'vegetation': 4,
        'not_vegetated': 5,
        'water': 6,
        'unclassified': 7,
        'cloud_medium_probability': 8,
        'cloud_high_probability': 9,
        'thin_cirrus': 10,
        'snow': 11
    }

    Parameters:
    -----------
    slc_data: 3D array
        Input array for computing the mask

    mask_values: list
        values to set to False in the mask

    erode_r : int
        Radius for eroding disk on the mask

    dilate_r : int
        Radius for dilating disk on the mask

    max_invalid_ratio : float
        Will set mask values to True, when they have an
        invalid_ratio > max_invalid_ratio

    Returns:
    --------
    mask : 3D bool array
        mask True for valid pixels, False for invalid

    obs : 2D int array
        number of valid observations (different from 0 in scl_data)

    invalid_before : 2D float array
        ratio of invalid obs before morphological operations

    invalid_after : 2D float array
        ratio of invalid obs after morphological operations
    """
    scl_data = scl_data.sel(band='SCL')

    mask_values = SCL_MASK_VALUES
    mask = da.isin(scl_data, mask_values)
    snow = scl_data == SCL_LEGEND['snow']
    water = scl_data == SCL_LEGEND['water']
    veg = scl_data == SCL_LEGEND['vegetation']
    notveg = scl_data == SCL_LEGEND['not_vegetated']

    ts_obs = scl_data != 0
    obs = ts_obs.sum(axis=0)

    ma_mask = (mask & ts_obs)
    invalid_before = ma_mask.sum(axis=0) / obs

    cover_snow = (snow & ts_obs).sum(axis=0) / obs
    cover_water = (water & ts_obs).sum(axis=0) / obs
    cover_veg = (veg & ts_obs).sum(axis=0) / obs
    cover_notveg = (notveg & ts_obs).sum(axis=0) / obs

    if (erode_r is not None) | (erode_r > 0):
        e = footprints.disk(erode_r)
        mask = da.stack([binary_erosion(m, e) for m in mask])

    if (dilate_r is not None) | (dilate_r > 0):
        d = footprints.disk(dilate_r)
        mask = da.stack([binary_dilation(m, d) for m in mask])

    ma_mask = (mask & ts_obs)
    invalid_after = ma_mask.sum(axis=0) / obs

    # invert values to have True for valid pixels and False for clouds
    mask = ~mask

    if max_invalid_ratio is not None:
        max_invalid_mask = invalid_after > max_invalid_ratio
        mask = mask | da.broadcast_to(max_invalid_mask, mask.shape)

    mask = scl_data.copy(data=mask)

    mask = mask.assign_coords(band='SCL')
    mask = mask.expand_dims('band', axis=1)

    return SCLMask(mask,
                   obs,
                   invalid_before,
                   invalid_after,
                   cover_snow,
                   cover_water,
                   cover_veg,
                   cover_notveg)
