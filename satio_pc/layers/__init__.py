"""
GeoJSONs layers that are used to load or visualize data.
"""
import json
import sys
if sys.version_info.minor < 7:
    from importlib_resources import open_text
else:
    from importlib.resources import open_text


_basenames = {'s2grid': 's2grid_bounds.geojson',
              's2grid_all': 's2grid_all_bounds.geojson',
              'countries': 'ne_countries_simple.geojson',
              'landsea': 'ne_landsea_simple.geojson'}

layers_description = {'s2grid': 'Sentinel-2 tiles grid GeoJSON',
                      's2grid_all': ('Sentinel-2 tiles grid GeoJSON '
                                     '(including tiles with S2 data.)'),
                      'countries': ('Natural Earth Vector layer of global '
                                    'countries at 10 m resolution.'),
                      'landsea': ('LandSea mask derived from Natural Earth '
                                  'coutries layer. Equivalent to '
                                  'countries.unary_union.simplify(0.01)')}


class SatioLayers:

    def __init__(self, layers=None, skip=[]):
        layers = layers if layers is not None else _basenames.keys()
        for k in layers:
            if k not in skip:
                exec(f'self.{k} = load("{k}")')


def load(*layers, skip=[]):
    """
    Providing only one 'layer_id' will return directly the geodataframe.

    Providing multiple layer ids (or none) will return an object where layers
    (all if none specified) can be accessed as attributes.

    Avalibale layers:

    {'s2grid': 'Sentinel-2 tiles grid GeoJSON',
    's2grid_all': ('Sentinel-2 tiles grid GeoJSON '
                    '(including tiles with S2 data.)'),
    'countries': ('Natural Earth Vector layer of global '
                'countries at 10 m resolution.'),
    'landsea': ('LandSea mask derived from Natural Earth '
                'coutries layer. Equivalent to '
                'countries.unary_union.simplify(0.01)'),
    'eco': ('Simplified (0.01) Resolve Ecoregions 2017. '
            'From https://ecoregions2017.appspot.com/'}
    """
    import geopandas as gpd

    if len(layers) == 1:
        with open_text('satio_pc.layers', _basenames[layers[0]]) as f:
            geojson_dict = json.load(f)
            gdf = gpd.GeoDataFrame.from_features(geojson_dict, crs=4326)

            if 's2grid' in layers[0]:
                # convert column of strings to
                gdf['bounds'] = gdf['bounds'].apply(eval)

            return gdf

    elif len(layers) == 0:
        return SatioLayers(None, skip=skip)
    else:
        return SatioLayers(layers, skip=skip)


def load_s2tile_windows(resolution):
    from rasterio.windows import Window  # NOQA Used to decode s2 tiles windows

    with open_text('satio_pc.layers', 'sentinel2_jp2_windows.json') as f:
        windows = json.load(f)

    return eval(windows[str(resolution)])
