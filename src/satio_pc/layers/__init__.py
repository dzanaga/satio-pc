import json
from importlib.resources import files, open_text


_basenames = {'s2grid': 's2grid_bounds.fgb',
              's2grid_all': 's2grid_all_bounds.fgb',
              'countries': 'ne_countries_simple.fgb',
              'landsea': 'ne_landsea_simple.fgb'}

layers_description = {'s2grid': 'Sentinel-2 tiles grid GeoJSON',
                      's2grid_all': ('Sentinel-2 tiles grid GeoJSON '
                                     '(including tiles with S2 data.)'),
                      'countries': ('Natural Earth Vector layer of global '
                                    'countries at 10 m resolution.'),
                      'landsea': ('LandSea mask derived from Natural Earth '
                                  'coutries layer. Equivalent to '
                                  'countries.unary_union.simplify(0.01)')}

def _fn(layer):
    if layer not in _basenames.keys():
        raise ValueError("The requested layer is not available. "
                         f"Available layers: {layers_description}")
    
    fn = files("satio_pc.layers").joinpath(_basenames[layer])
    return fn

def load(layer, mask=None, bbox=None):
    """
    Load layer. `mask` and `bbox` can be a shapely geometry or a
    (xmin, ymin, xmax, ymax) lat lon tuple to restrict the layer reading.
    
    Avalibale layers:

    {'s2grid': 'Sentinel-2 tiles grid GeoJSON',
    's2grid_all': ('Sentinel-2 tiles grid GeoJSON '
                    '(including tiles with S2 data.)'),
    'countries': ('Natural Earth Vector layer of global '
                'countries at 10 m resolution.'),
    'landsea': ('LandSea mask derived from Natural Earth '
                'coutries layer. Equivalent to '
                'countries.unary_union.simplify(0.01)')}
    """
    import geopandas as gpd

    gdf = gpd.read_file(_fn(layer), mask=mask, bbox=bbox)

    if 's2grid' in layer:
        # convert column of strings to
        gdf['bounds'] = gdf['bounds'].apply(eval)

    return gdf


def load_s2tile_windows(resolution):
    from rasterio.windows import Window  # NOQA Used to decode s2 tiles windows

    with open_text('satio_pc.layers', 'sentinel2_jp2_windows.json') as f:
        windows = json.load(f)

    return eval(windows[str(resolution)])
