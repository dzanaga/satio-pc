import numpy as np


WL_B02, FWHM_B02 = (0.4927 + 0.4923) / 2, (0.065 + 0.065) / 2
WL_B03, FWHM_B03 = (0.5598 + 0.5589) / 2, (0.035 + 0.035) / 2
WL_B04, FWHM_B04 = (0.6646 + 0.6649) / 2, (0.030 + 0.031) / 2
WL_B08, FWHM_B08 = (0.8328 + 0.8329) / 2, (0.105 + 0.104) / 2
WL_B11, FWHM_B11 = (1.6137 + 1.6104) / 2, (0.090 + 0.094) / 2
WL_B12, FWHM_B12 = (2.2024 + 2.1857) / 2, (0.174 + 0.184) / 2


def norm_diff(arr1, arr2):
    """Returns the normalized difference of two bands"""
    return (arr1 - arr2) / (arr1 + arr2)


class IndicesRegistry:
    ...


s2 = IndicesRegistry()


class S2Indices:

    _clip = True

    def __init_subclass__(cls):

        if not hasattr(cls, 'name'):
            raise ValueError("Subclasses must have a 'name' attribute.")

        setattr(s2, cls.name, cls())

    def clip(self, arr):

        if self._clip:
            vmin, vmax = self.values_range
        else:
            vmin = vmax = np.nan

        arr[arr < vmin] = vmin
        arr[arr > vmax] = vmax

        return arr


class AUC(S2Indices):

    name = 'auc'
    bands = 'B02', 'B03', 'B04', 'B08', 'B11', 'B12'
    values_range = 0, 0.506

    def __call__(self, B02, B03, B04, B08, B11, B12):

        arr = (B02 * FWHM_B02 + B03 * FWHM_B03 +
               B04 * FWHM_B04 + B08 * FWHM_B08 +
               B11 * FWHM_B11 + B12 * FWHM_B12)

        return self.clip(arr)


class NAUC(S2Indices):

    name = 'nauc'
    bands = 'B02', 'B03', 'B04', 'B08', 'B11', 'B12'
    values_range = 0, 0.506

    def __call__(self, B02, B03, B04, B08, B11, B12):

        min_ref = np.fmin(np.fmin(np.fmin(B02, B03),
                                  np.fmin(B04, B08)
                                  ),
                          np.fmin(B11, B12)
                          )

        arr = ((B02 - min_ref) * FWHM_B02
               + (B03 - min_ref) * FWHM_B03
               + (B04 - min_ref) * FWHM_B04
               + (B08 - min_ref) * FWHM_B08
               + (B11 - min_ref) * FWHM_B11
               + (B12 - min_ref) * FWHM_B12)

        return self.clip(arr)


class NDVI(S2Indices):

    name = 'ndvi'
    bands = 'B08', 'B04'
    values_range = -1, 1

    def __call__(self, B08, B04):
        arr = norm_diff(B08, B04)
        return self.clip(arr)


class NDMI(S2Indices):

    name = 'ndvi'
    bands = 'B08', 'B04'
    values_range = -1, 1

    def __call__(self, B08, B04):
        arr = norm_diff(B08, B04)
        return self.clip(arr)


class NBR(S2Indices):

    name = 'ndvi'
    bands = 'B08', 'B04'
    values_range = -1, 1

    def __call__(self, B08, B04):
        arr = norm_diff(B08, B04)
        return self.clip(arr)

        # if rsi_name in ['ndvi', 'ndmi', 'nbr', 'nbr2', 'ndwi', 'ndgi',
        #                 'ndre1', 'ndre2', 'ndre3', 'ndre4', 'ndre5',
        #                 'mndwi']:
