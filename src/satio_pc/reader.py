import xml.etree.ElementTree as ET

import requests
import numpy as np
import xarray as xr
import rasterio.windows
from rasterio.enums import Resampling
from loguru import logger
from stackstac.stack import items_to_plain, prepare_items, to_coords, to_attrs
from stackstac.to_dask import asset_table_to_reader_and_window
from stackstac.rio_reader import AutoParallelRioReader

from satio_pc import parallelize
from satio_pc.sentinel2 import query_l2a_items


def get_view_angles(item):
    """Parse GRANULE xml for the mean view azimuth and zenith angles.
    Returns 2 dictionaries azimuth and zenith of the angles for each band"""
    
    url = item.assets['granule-metadata'].href
    
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors

    root = ET.fromstring(response.content)

    # Extract the namespace from the root tag
    namespace = None
    for attr_name, attr_value in root.attrib.items():
        namespace = '{' + attr_value.split()[0] + '}'
    
    azimuth = {}
    zenith = {}
    
    el = root.find(".//Mean_Viewing_Incidence_Angle_List",
                   namespaces={"n1": namespace})
    for angle in el.findall('Mean_Viewing_Incidence_Angle'):
        band_id = angle.attrib['bandId']
        zenith[band_id] = float(angle.find('ZENITH_ANGLE').text)
        azimuth[band_id] = float(angle.find('AZIMUTH_ANGLE').text)
     
    return azimuth, zenith


def get_mean_view_angles(item):
    """Compute mean of the angles across all bands"""
    azimuth, zenith = get_view_angles(item)
    mean_azimuth = float(np.array(list(azimuth.values())).mean())
    mean_zenith = float(np.array(list(zenith.values())).mean())
    return mean_azimuth, mean_zenith


def add_mean_view_angles(items, workers=8):
    """Add the mean azimuth and zenith angles to the STAC items"""
    
    # metadata is parsed in a ThreadPool to reduce network latency
    items_azimuth, items_zenith = list(zip(*parallelize(
        get_mean_view_angles,
        items,
        max_workers=workers)))

    for item, azimuth, zenith in zip(items, items_azimuth, items_zenith):
        item.properties['s2:mean_view_azimuth'] = azimuth
        item.properties['s2:mean_view_zenith'] = zenith
        
    return items


def load_reader_table_items(reader_table, spec, fill_value, dtype):
    current_window = rasterio.windows.from_bounds(
        *spec.bounds, transform=spec.transform)

    output = np.broadcast_to(
        np.array(fill_value, dtype),
        reader_table.shape + (int(current_window.height),
                              int(current_window.width)),
    )

    all_empty: bool = True
    for index, entry in np.ndenumerate(reader_table):
        if entry:
            reader, asset_window = entry
            data = reader.read(current_window)

            if all_empty:
                # Turn `output` from a broadcast-trick array to a real array
                if (
                    np.isnan(data)
                    if np.isnan(fill_value)
                    else np.equal(data, fill_value)
                ).all():
                    # Unless the data we just read is all empty anyway
                    continue
                output = np.array(output)
                all_empty = False

            output[index] = data

    return output


def load_reader_table_items_thread_pool(reader_table,
                                        spec,
                                        fill_value,
                                        dtype,
                                        max_workers=20):
    current_window = rasterio.windows.from_bounds(
        *spec.bounds, transform=spec.transform)

    output = np.broadcast_to(
        np.array(fill_value, dtype),
        reader_table.shape + (int(current_window.height),
                              int(current_window.width)),
    )
    output = np.array(output)

    def _read(index_entry):
        # for index, entry in np.ndenumerate(reader_table):
        index, entry = index_entry
        # if entry:
        #     print(entry)
        #     return
        reader, asset_window = entry

        data = reader.read(current_window)
        output[index] = data

    _ = parallelize(
        _read,
        list(np.ndenumerate(reader_table)),
        max_workers=max_workers,
        progressbar=False)

    return output


def load_items(
    items,
    assets,
    bounds,
    epsg,
    resolution,
    dtype,
    xy_coords='center',
    fill_value=0,
    resampling=Resampling.nearest,
    max_workers=20,
):

    plain_items = items_to_plain(items)
    plain_items = sorted(
        plain_items,
        key=lambda item: item["properties"].get("datetime", "") or "",
        reverse=False,
    )

    asset_table, spec, asset_ids, plain_items = prepare_items(
        plain_items,
        assets=assets,
        epsg=epsg,
        resolution=resolution,
        bounds=bounds,
        bounds_latlon=None,
        snap_bounds=True,
    )

    reader_table = asset_table_to_reader_and_window(
        asset_table,
        spec,
        resampling,
        dtype=dtype,
        fill_value=fill_value,
        rescale=False,
        gdal_env=None,
        errors_as_nodata=(),
        reader=AutoParallelRioReader)

    if max_workers == -1:
        arr = load_reader_table_items(reader_table, spec, fill_value, dtype)
    else:
        arr = load_reader_table_items_thread_pool(
            reader_table, spec, fill_value, dtype, max_workers)

    return xr.DataArray(
        arr,
        *to_coords(
            plain_items,
            asset_ids,
            spec,
            xy_coords=xy_coords,
            properties=True,
            band_coords=True,
        ),
        attrs=to_attrs(spec),
        name="stac-data",
    )


class S2TileReader:

    def __init__(self,
                 tile,
                 start_date,
                 end_date,
                 max_cloud_cover=90,
                 filter_corrupted=True,
                 add_view_angles=True):

        self._tile = tile
        self._start_date = start_date
        self._end_date = end_date

        self._max_cloud_cover = max_cloud_cover
        self._filter_corrupted = filter_corrupted

        self._items = None
        self._add_view_angles = add_view_angles

    @property
    def items(self):
        if self._items is None:
            self._items = query_l2a_items(self._tile,
                                          self._start_date,
                                          self._end_date,
                                          self._max_cloud_cover,
                                          self._filter_corrupted)

            self._items.items = sorted(self._items.items,
                                       key=lambda item: item.datetime)

            if self._add_view_angles:
                logger.debug("Parsing mean_view_azimuth and mean_view_zenith "
                             "angles properties.")
                self._items = add_mean_view_angles(self._items)
                
        return self._items

    def assets(self, bands):

        assets_10m = set(['B02', 'B03', 'B04', 'B08'])
        assets_20m = set(['B05', 'B06', 'B07', 'B8A', 'B11', 'B12'])
        assets_60m = set(['B01', 'B09', 'WVP'])

        sbands = set(bands)

        dtype = np.uint16

        if len(sbands & assets_10m):
            resolution = 10
        elif len(sbands & assets_20m):
            resolution = 20
        elif len(sbands & assets_60m):
            resolution = 60
        elif bands == ['SCL']:
            dtype = np.uint8
            resolution = 20
        else:
            # raise ValueError(
            #     f"Bands is a mix of resolutions or not recognized: {bands}")
            logger.warning(f"Bands is a mix of resolutions or"
                           f"not recognized: {bands}. "
                           "Suggesting 10m resolution.")

        return bands, dtype, resolution

    def read(self,
             bounds,
             epsg,
             bands,
             resolution=None,
             max_workers=20,
             resampling=None):

        assets, dtype, native_resolution = self.assets(bands)

        if resolution is None:
            resolution = native_resolution
            
        if resampling is None:
            resampling = Resampling.nearest

        darr = load_items(
            self.items,
            assets,
            bounds,
            f'EPSG:{epsg}',
            resolution,
            dtype,
            xy_coords='center',
            fill_value=0,
            resampling=resampling,
            max_workers=max_workers,
        )
        return darr
