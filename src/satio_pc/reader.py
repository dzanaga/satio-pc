import numpy as np
import xarray as xr
import rasterio.windows
from rasterio.enums import Resampling

from stackstac.stack import items_to_plain, prepare_items, to_coords, to_attrs
from stackstac.to_dask import asset_table_to_reader_and_window
from stackstac.rio_reader import AutoParallelRioReader

from satio_pc.utils import parallelize
from satio_pc.sentinel2 import query_l2a_items


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
                 filter_corrupted=True):

        self._tile = tile
        self._start_date = start_date
        self._end_date = end_date

        self._max_cloud_cover = max_cloud_cover
        self._filter_corrupted = filter_corrupted

        self._items = None

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

        return self._items

    def assets(self, bands):

        assets_10m = set(['B02', 'B03', 'B04', 'B08'])
        assets_20m = set(['B05', 'B06', 'B07', 'B8A', 'B11', 'B12'])
        assets_60m = set(['B01', 'B09'])

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
            raise ValueError(
                f"Bands is a mix of resolutions or not recognized: {bands}")

        return bands, dtype, resolution

    def read(self,
             bounds,
             epsg,
             bands,
             max_workers=20):

        assets, dtype, resolution = self.assets(bands)

        darr = load_items(
            self.items,
            assets,
            bounds,
            f'EPSG:{epsg}',
            resolution,
            dtype,
            xy_coords='center',
            fill_value=0,
            resampling=Resampling.nearest,
            max_workers=max_workers,
        )
        return darr
