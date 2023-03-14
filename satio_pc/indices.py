"""
Vegetation indices:
NDVI = Normalized Difference Vegetation Index (Rouse et al., 1974) – NDVI is used to quantify vegetation greenness and is useful in understanding vegetation density and assessing changes in plant health.
EVI = Enhanced Vegetation Index (Huete et al., 2002) - EVI is similar to Normalized Difference Vegetation Index (NDVI) and can be used to quantify vegetation greenness. However, EVI corrects for some atmospheric conditions and canopy background noise and is more sensitive in areas with dense vegetation.
NIRV = Near-Infrared Reflectance of Vegetation (Badgley et al., 2017) – represents the proportion of pixel reflectance attributable to vegetation in the pixel
NDWI = Normalized Difference Water Index (McFeeters et al., 1996) – related to water content in water bodies
NDGI = Normalized Difference Greenness Index (or more common name is NGRDI or Normalized difference green/red index)(Tucker, 1979) – related to greenness
NDMI = Normalized Difference Moisture Index (Gao et al., 1996) – related to the water content of leaves
NBR = Normalized Burn Ratio (Garcia et al., 1991) - NBR is often used to identify burned areas and provide a measure of burn severity.
NBR2 = Normalized Burn Ratio2 (Garcia et al., 1991) - NBR2 modifies the Normalized Burn Ratio (NBR) to highlight water sensitivity in vegetation.
REP = Red Edge Position (Curran et al., 1995) -  sensitive to changes in chlorophyll concentration
ANIR = Angle at Near-Infrared (Khanna et al., 2007) – suited to detect to dry plant matter in the presence of soil and  green vegetation cover
NDRE2 = Normalized Difference Red Edge index – based on Sentinel-2 red edge band 6, chlorophyll content  (Gitelson & Merzlyak, 1994)
NDRE3 = Normalized Difference Red Edge index – based on Sentinel-2 red edge band 7, chlorophyll content (Gitelson & Merzlyak, 1994)

Badgley, G., Field, C. B., & Berry, J. A. (2017). Canopy near-infrared reflectance and terrestrial photosynthesis. Science advances, 3(3), e1602244.

Curran, P. J., Windham, W. R., & Gholz, H. L. (1995). Exploring the relationship between reflectance red edge and chlorophyll concentration in slash pine leaves. Tree physiology, 15(3), 203-206.

Gao, B. C. (1996). NDWI—A normalized difference water index for remote sensing of vegetation liquid water from space. Remote sensing of environment, 58(3), 257-266.

García, M. L., & Caselles, V. (1991). Mapping burns and natural reforestation using Thematic Mapper data. Geocarto International, 6(1), 31-37.

Gitelson, A., & Merzlyak, M. N. (1994). Spectral reflectance changes associated with autumn senescence of Aesculus hippocastanum L. and Acer platanoides L. leaves. Spectral features and relation to chlorophyll estimation. Journal of plant physiology, 143(3), 286-292.

Huete, A., Didan, K., Miura, T., Rodriguez, E. P., Gao, X., & Ferreira, L. G. (2002). Overview of the radiometric and biophysical performance of the MODIS vegetation indices. Remote sensing of environment, 83(1-2), 195-213.

Khanna, S., Palacios-Orueta, A., Whiting, M. L., Ustin, S. L., Riaño, D., & Litago, J. (2007). Development of angle indexes for soil moisture estimation, dry matter detection and land-cover discrimination. Remote sensing of environment, 109(2), 154-165.

McFeeters, S. K. (1996). The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features. International journal of remote sensing, 17(7), 1425-1432.

Rouse, J. W., Haas, R. H., Schell, J. A., & Deering, D. W. (1974). Monitoring vegetation systems in the Great Plains with ERTS. NASA special publication, 351(1974), 309.

Tucker, C. J. (1979). Red and photographic infrared linear combinations for monitoring vegetation. Remote sensing of Environment, 8(2), 127-150.
"""

import numpy as np

NODATA_VALUE = -2**15

RSI_META_S2 = {
    'ndvi': {
        'bands': ['B08', 'B04'],
        'range': [-1, 1],
        'clamp_range': False,
        'scale': 2000,
        'native_res': 10},

    # NDWI (Gao, 1996)
    'ndmi': {'bands': ['B08', 'B11'],
             'range': [-1, 1],
             'clamp_range': False,
             'scale': 10000,
             'native_res': 20},

    'nbr': {'bands': ['B08', 'B12'],
            'range': [-1, 1],
            'clamp_range': False,
            'scale': 10000,
            'native_res': 20},

    'nbr2': {'bands': ['B11', 'B12'],
             'range': [-3, 3],
             'clamp_range': False,
             'scale': 10000,
             'native_res': 20},

    'evi': {'bands': ['B08', 'B04', 'B02'],
            'range': [-3, 3],
            'clamp_range': False,
            'scale': 10000,
            'native_res': 10},

    'evi2': {'bands': ['B08', 'B04'],
             'range': [-3, 3],
             'clamp_range': False,
             'scale': 10000,
             'native_res': 10},

    'savi': {'bands': ['B08', 'B04'],
             'range': [-3, 3],
             'clamp_range': False,
             'scale': 10000,
             'native_res': 10},

    'sipi': {'bands': ['B08', 'B01', 'B04'],
             'range': [-10, 10],
             'clamp_range': False,
             'scale': 1000,
             'native_res': 60},

    'hsvh': {'bands': ['B04', 'B03', 'B02'],
             'range': [0, 1],
             'clamp_range': False,
             'scale': 18000,
             'native_res': 10},

    'hsvv': {'bands': ['B04', 'B03', 'B02'],
             'range': [0, 1],
             'clamp_range': False,
             'scale': 2000,
             'native_res': 10},

    'hsv': {'bands': ['B04', 'B03', 'B02'],
            'range': [0, 1],
            'clamp_range': False,
            'scale': 2000,
            'native_res': 10,
            'output_bands': ['hsvh', 'hsvv']},

    'rep': {'bands': ['B04', 'B07', 'B05', 'B06'],
            'range': [500, 900],
            'clamp_range': False,
            'scale': 10,
            'native_res': 20},

    'anir': {'bands': ['B04', 'B08', 'B11'],
             'range': [0, 1],
             'clamp_range': False,
             'scale': 1,
             'native_res': 20},

    'nirv': {'bands': ['B08', 'B04'],
             'range': [-1, 1],
             'clamp_range': False,
             'scale': 10000,
             'native_res': 10},

    'auc': {'bands': ['B02', 'B04', 'B08', 'B11'],
            'range': [0, 1],
            'clamp_range': False,
            'scale': 20000,
            'native_res': 20},

    'nauc': {'bands': ['B02', 'B04', 'B08', 'B11'],
             'range': [0, 1],
             'clamp_range': False,
             'scale': 20000,
             'native_res': 20},

    # ndwi (mcFeeters)
    'ndwi': {'bands': ['B03', 'B08'],
             'range': [-1, 1],
             'clamp_range': False,
             'scale': 1,
             'native_res': 10},

    # modified NDWI (Xu, 2006)
    'mndwi': {'bands': ['B03', 'B11'],
              'range': [-1, 1],
              'clamp_range': False,
              'scale': 1,
              'native_res': 20},

    # normalized difference greenness index
    'ndgi': {'bands': ['B03', 'B04'],
             'range': [-1, 1],
             'clamp_range': False,
             'scale': 1,
             'native_res': 10},

    # bare soil index
    'bsi': {'bands': ['B02', 'B04', 'B08', 'B11'],
            'range': [-1, 1],
            'clamp_range': False,
            'scale': 1,
            'native_res': 20
            },

    # brightness (as defined in sen2agri)
    'brightness': {'bands': ['B03', 'B04', 'B08', 'B11'],
                   'range': [0, 1],
                   'clamp_range': False,
                   'scale': 1,
                   'native_res': 20},

    # series of normalized difference rede edge indices
    'ndre1': {'bands': ['B08', 'B05'],
              'range': [-1, 1],
              'clamp_range': False,
              'scale': 1,
              'native_res': 20},

    'ndre2': {'bands': ['B08', 'B06'],
              'range': [-1, 1],
              'clamp_range': False,
              'scale': 1,
              'native_res': 20},

    'ndre3': {'bands': ['B08', 'B07'],
              'range': [-1, 1],
              'clamp_range': False,
              'scale': 1,
              'native_res': 20},

    'ndre4': {'bands': ['B06', 'B05'],
              'range': [-1, 1],
              'clamp_range': False,
              'scale': 1,
              'native_res': 20},

    'ndre5': {'bands': ['B07', 'B05'],
              'range': [-1, 1],
              'clamp_range': False,
              'scale': 1,
              'native_res': 20}}

RSI_META_S1 = {
    'vh_vv': {
        'bands': ['VH', 'VV'],
        'range': [-20, 0],
        'clamp_range': False,
        'scale': 1,
        'native_res': 20},
    'rvi': {
        'bands': ['VH', 'VV'],
        'range': [0, 2],
        'clamp_range': False,
        'scale': 1,
        'native_res': 20}}

RSI_META_AgERA5 = {
    'et0': {
        'bands': ['temperature_min', 'temperature_max',
                  'solar_radiation_flux', 'vapour_pressure',
                  'wind_speed', 'elev', 'doy', 'lat'],
        'range': [0, 15],
        'clamp_range': False,
        'scale': 1,
        'native_res': 100,
        'prior': True},
    'et': {
        'bands': ['et0'],
        'range': [0, 15],
        'clamp_range': False,
        'scale': 1,
        'native_res': 20},
    'prdef': {
        'bands': ['et0', 'precipitation_flux'],
        'range': [-15, 15],
        'clamp_range': False,
        'scale': 1,
        'native_res': 20}
}

RSI_META_L8 = {
    'ndvi': {
        'bands': ['B05', 'B04'],
        'range': [-1, 1],
        'clamp_range': False,
        'scale': 1,
        'native_res': 10},
}

RSI_META = {'S2': RSI_META_S2,
            'S1': RSI_META_S1,
            'AgERA5': RSI_META_AgERA5,
            'L8': RSI_META_L8}

WL_B02, FWHM_B02 = 0.4924, 0.098
WL_B04, FWHM_B04 = 0.6646, (0.039 + 0.038) / 2
WL_B08, FWHM_B08 = 0.8328, (0.145 + 0.133) / 2
WL_B11, FWHM_B11 = 1.610, (0.143 + 0.141) / 2

# supported_vis = VI_META.keys()


def get_rsi_function(rsi_name, meta=None):
    """
    Derive RSI function either from its name or from meta
    The rsi_name argument suffices for indices defined within satio
    If a custom RSI is required, the function should be defined in
    meta, as a callable under the 'func' key.

    :param rsi_name: string
    :param meta: optional dictionary containing a 'func' key

    """
    if meta is not None and 'func' in meta.keys():
        f = meta['func']
    else:
        if rsi_name in ['ndvi', 'ndmi', 'nbr', 'nbr2', 'ndwi', 'ndgi',
                        'ndre1', 'ndre2', 'ndre3', 'ndre4', 'ndre5',
                        'mndwi']:
            f = norm_diff
        else:
            f = eval(rsi_name)
    return f


def norm_diff(arr1, arr2):
    """Returns the normalized difference of two bands"""
    return (arr1 - arr2) / (arr1 + arr2)


def evi(B08, B04, B02):
    return 2.5 * (B08 - B04) / (B08 + 6.0 * B04 - 7.5 * B02 + 1.0)


def evi2(B08, B04):
    return 2.5 * (B08 - B04) / (B08 + 2.4 * B04 + 1.0)


def savi(B08, B04):
    L = 0.428
    return (B08 - B04) / (B08 + B04 + L) * (1.0 + L)


def sipi(B08, B01, B04):
    return (B08 - B01) / (B08 - B04)


def hsv(B04, B03, B02):
    """Returns hsv 3d array from RGB bands"""
    nodata = np.isnan(B04)
    h, v = get_hsv_hue_value(B04, B03, B02)
    h[nodata] = np.nan
    v[nodata] = np.nan
    return np.array([h, v])


def hsvh(B04, B03, B02):
    hv = hsv(B04, B03, B02)
    return hv[0]


def hsvv(B04, B03, B02):
    hv = hsv(B04, B03, B02)
    return hv[1]


def rep(B04, B07, B05, B06):
    return 700 + 40 * ((((B04 + B07) / 2) - B05) / (B06 - B05))


def anir(B04, B08, B11):
    a = np.sqrt(np.square(WL_B08 - WL_B04) + np.square(B08 - B04))
    b = np.sqrt(np.square(WL_B11 - WL_B08) + np.square(B11 - B08))
    c = np.sqrt(np.square(WL_B11 - WL_B04) + np.square(B11 - B04))

    # calculate angle with NIR as reference (ANIR)
    site_length = (np.square(a) + np.square(b) - np.square(c)) / (2 * a * b)
    site_length[site_length < -1] = -1
    site_length[site_length > 1] = 1

    return 1. / np.pi * np.arccos(site_length)


def nirv(B08, B04):
    return ((B08 - B04 / B08 + B04) - 0.08) * B08


def auc(B02, B04, B08, B11):
    return B02 * FWHM_B02 + B04 * FWHM_B04 + B08 * FWHM_B08 + B11 * FWHM_B11


def nauc(B02, B04, B08, B11):
    min_ref = np.fmin(np.fmin(B02, B04), np.fmin(B08, B11))
    return ((B02 - min_ref) * FWHM_B02
            + (B04 - min_ref) * FWHM_B04
            + (B08 - min_ref) * FWHM_B08
            + (B11 - min_ref) * FWHM_B11)


def get_hsv_timeseries(r, g, b):

    h, s, v = np.zeros(r.shape), np.zeros(r.shape), np.zeros(r.shape)

    rgb = np.array([r, g, b])
    mx = np.nanmax(rgb, 0)
    mn = np.nanmin(rgb, 0)

    diff = mx - mn

    h[mx == mn] = 0
    h[mx == r] = ((60 * ((g - b) / diff) + 360) % 360)[mx == r]
    h[mx == g] = ((60 * ((b - r) / diff) + 360) % 360)[mx == g]
    h[mx == b] = ((60 * ((r - g) / diff) + 360) % 360)[mx == b]
    h = h / 360

    s = diff / mx
    s[mx == 0] = 0

    v = mx

    return h, s, v


def get_hsv_hue_value(r, g, b):
    h = np.zeros(r.shape, dtype=np.float32)

    rgb = np.array([r, g, b])
    mx = np.nanmax(rgb, 0)
    mn = np.nanmin(rgb, 0)

    diff = mx - mn

    h[mx == mn] = 0
    with np.errstate(divide='ignore', invalid='ignore'):  # type: ignore
        h[mx == r] = ((60 * ((g - b) / diff) + 360) % 360)[mx == r]
        h[mx == g] = ((60 * ((b - r) / diff) + 360) % 360)[mx == g]
        h[mx == b] = ((60 * ((r - g) / diff) + 360) % 360)[mx == b]

    h = h / 360
    v = mx

    return h, v


def brightness(B03, B04, B08, B11):
    return np.sqrt(np.power(B03, 2) + np.power(B04, 2)
                   + np.power(B08, 2) + np.power(B11, 2))


def _to_db(pwr):
    '''
    Helper function to transform dB to power units
    '''
    return 10 * np.log10(pwr)


def _to_pwr(db):
    '''
    Helper function to transform power to dB units
    '''
    return np.power(10, db / 10)


def vh_vv(VH, VV):
    """Function to calculate VH/VV ratio in dB

    Args:
        VH: VH time series in decibels
        VV: VV time series in decibels

    Returns:
        ndarray: VH/VV ratio in decibels
    """
    # Calculte ratio using logarithm rules
    return VH - VV


def rvi(VH, VV):
    """Function to calculate radar vegetation index

    Args:
        VH: VH time series in decibels
        VV: VV time series in decibels

    Returns:
        ndarray: RVI [dimensionless]
    """
    VH = _to_pwr(VH)
    VV = _to_pwr(VV)

    return (4 * VH) / (VV + VH)


def bsi(B02, B04, B08, B11):
    """Function to calculate bare soil index
    """

    bsi = ((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02))

    return bsi
