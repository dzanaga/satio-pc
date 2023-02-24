from typing import List

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.geometry import mapping, shape
from fiona.transform import transform_geom
from pyproj.crs import CRS
import rasterio
import rasterio.mask

import satio_pc
from satio_pc.layers import load_s2tile_windows


def tile_to_epsg(tile):
    row = tile[2]
    zone = tile[:2]

    if row in 'CDEFGHJKLM':
        hemisphere = 'S'
    elif row in 'NPQRSTUVWX':
        hemisphere = 'N'
    else:
        raise ValueError(f"Unrecognized UTM zone '{zone}'.")

    utm = zone + hemisphere
    return utm_to_epsg(utm)


def utm_to_epsg(utm):
    utm = utm.upper()
    sud = 1 if utm[-1] == 'S' else 0
    zone = int(utm[:-1])
    epsg = 32600 + sud * 100 + zone
    return epsg


def get_tile_blocks(tile, s2grid=None, resolution=10):

    if s2grid is None:
        s2grid = satio_pc.layers.load('s2grid')

    width = heigth = {10: 10980,
                      20: 5490,
                      60: 1830}[resolution]

    epsg = tile_to_epsg(tile)
    if 'bounds' in s2grid.columns:
        tile_bounds = s2grid.loc[s2grid.tile == tile, 'bounds'].iloc[0]
    else:
        tile_bounds = s2grid[s2grid.tile == tile].to_crs(
            epsg=epsg).bounds.values[0].round().tolist()
    tile_transform = rasterio.transform.from_bounds(
        *tile_bounds, width, heigth)

    windows_tuples = load_s2tile_windows(resolution)

    polygons = []
    for t in windows_tuples:
        w = t[1]
        xmin, ymax = tile_transform * (w.col_off, w.row_off)
        xmax, ymin = tile_transform * \
            (w.col_off + w.width, w.row_off + w.height)

        polygons.append(Polygon.from_bounds(xmin, ymin, xmax, ymax))

    return gpd.GeoSeries(polygons, crs=CRS.from_epsg(epsg))


def get_blocks_gdf(tiles, s2grid=None, resolution=10):
    if s2grid is None:
        s2grid = satio_pc.layers.load('s2grid')

    tiles_blocks = []
    for t in tiles:
        tblocks = get_tile_blocks(t, s2grid, resolution=resolution)
        tblocks_ll = tblocks.to_crs(epsg=4326)
        tiles_blocks += [{'tile': t,
                          'bounds': tuple(np.round(
                              np.array(b.bounds) / resolution).astype(int)
                              * resolution),
                          'geometry': tblocks_ll.iloc[i],
                          'area': b.area,
                          'epsg': tile_to_epsg(t),
                          'block_id': i}
                         for i, b in enumerate(tblocks)]
    df = gpd.GeoDataFrame(tiles_blocks, crs=CRS.from_epsg(4326))

    return df


def get_blocks_gdf_antimeridian(tiles, s2grid=None, resolution=10):
    if s2grid is None:
        s2grid = satio_pc.layers.load('s2grid')

    tiles_blocks = []
    for t in tiles:
        tblocks = get_tile_blocks(t, s2grid, resolution=resolution)
        tblocks_ll = fiona_transform(tblocks.to_frame('geometry'),
                                     dst_epsg=4326)

        tiles_blocks += [{'tile': t,
                          'bounds': tuple(np.round(
                              np.array(b.bounds) / resolution).astype(int)
                              * resolution),
                          'geometry': tblocks_ll.iloc[i].geometry,
                          'area': b.area,
                          'epsg': tile_to_epsg(t),
                          'block_id': i}
                         for i, b in enumerate(tblocks)]
    df = gpd.GeoDataFrame(tiles_blocks, crs=CRS.from_epsg(4326))

    return df


def buffer_bounds(bounds, buffer):
    bounds = np.array(bounds)
    bounds += np.array([-buffer, -buffer, buffer, buffer])
    return bounds.tolist()


def clip_to_global_bbox(df):
    bbox = gpd.GeoSeries([Polygon([(-180, 90), (180, 90),
                                   (180, -90), (-180, -90)])])
    dfbbox = gpd.GeoDataFrame({'geometry': bbox,
                               'gbbox': 0}, crs=CRS.from_epsg(4326))
    dfint = gpd.tools.overlay(df, dfbbox)
    return dfint


def fiona_transform(df, dst_crs=None, dst_epsg=None):
    if dst_epsg is not None:
        dst_crs = CRS.from_epsg(dst_epsg)

    if not isinstance(dst_crs, str):
        dst_crs = dst_crs.to_string()

    src_crs = df.crs.to_string()

    def f(x): return fiona_transformer(src_crs, dst_crs, x)

    tdf = df.set_geometry(df.geometry.apply(f))
    tdf.crs = CRS.from_string(dst_crs)
    return tdf


def fiona_transformer(src_crs, dst_crs, geom):
    fi_geom = transform_geom(src_crs=src_crs,
                             dst_crs=dst_crs,
                             geom=mapping(geom),
                             antimeridian_cutting=True)
    return shape(fi_geom)


def get_latlon_grid(deg_resolution: int = 1,
                    sjoin_layers: List[gpd.GeoDataFrame] = None):
    """
    Genearte a gloabl lat lon grid with conventional cell names.

    If a list of `sjoin_layers` is provided they will be intersected to
    reduce the grid. (For example providing the landsea layer it will output
    a grid of cells only covering landmass)
    """
    if (deg_resolution < 1) or not isinstance(int):
        raise ValueError('`deg_resolution` should be an integer greater '
                         'than 1.')

    sjoin_layers = sjoin_layers or []

    xdeg = ydeg = deg_resolution

    x = np.arange(-180, 180, xdeg)
    y = np.arange(-90, 90, ydeg)

    grid_origins = [(x0, y0) for y0 in y for x0 in x]
    polygons = [Polygon.from_bounds(x0, y0, x0 + xdeg, y0 + ydeg)
                for x0, y0 in grid_origins]

    ll_tile = []
    for x0, y0 in grid_origins:
        a = 'E' if x0 >= 0 else 'W'
        b = 'N' if y0 >= 0 else 'S'
        ll_tile.append(f'{b}{abs(y0):02d}{a}{abs(x0):03d}')

    grid = gpd.GeoDataFrame(ll_tile, columns=['ll_tile'],
                            geometry=polygons, crs=4326)

    for layer in sjoin_layers:
        grid = gpd.sjoin(grid, layer)
        grid = grid.drop_duplicates('ll_tile')

    return grid


def epsg_point_bounds(p, epsg, dst_epsg,
                      box_pixels_shape_xy,
                      resolution=20):
    """Returns bounds and epsg for a box around the closest pixel corner
    to the given point.
    e.g. starting from a lat lon point, get an epsg box around it"""

    src_epsg = f'EPSG:{epsg}'
    dst_epsg = f'EPSG:{dst_epsg}'

    putm = fiona_transformer(src_epsg, dst_epsg, p)

    box_x = box_pixels_shape_xy[0] * resolution
    box_y = box_pixels_shape_xy[1] * resolution

    # tloc origin
    ox = round((putm.x - box_x/2) / resolution)*resolution
    oy = round((putm.y + box_y/2) / resolution)*resolution

    bounds = ox, oy - box_y, ox + box_x, oy

    return bounds
