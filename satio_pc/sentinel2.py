import tempfile
import xml.etree.ElementTree as ET
import datetime

import requests
import numpy as np
import xarray as xr
import dask.array as da
from loguru import logger

import satio_pc  # noqa register extension
from satio_pc import parallelize
from satio_pc.preprocessing.timer import FeaturesTimer


BANDS_10M = ['B02', 'B03', 'B04', 'B08']
BANDS_20M = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'SCL']
BANDS_60M = ['B01', 'B09', 'B10']

BANDS_RESOLUTION = dict(zip(BANDS_10M + BANDS_20M + BANDS_60M,
                            [10] * len(BANDS_10M) +
                            [20] * len(BANDS_20M) +
                            [60] * len(BANDS_60M)))


def get_quality_values(item):
    meta_url = item.assets['product-metadata'].href
    response = requests.get(meta_url)
    xml_content = response.content

    root = ET.fromstring(xml_content)

    # find the Quality_Inspections element and get all the quality_check sub-elements
    quality_checks = root.find(
        './/Quality_Inspections').findall('quality_check')

    # create a dictionary to store the quality values
    quality_values = {}

    # iterate over the quality_check elements and extract the checkType and PASSED/FAILED value
    for qc in quality_checks:
        check_type = qc.attrib['checkType']
        value = qc.text
        quality_values[check_type] = value

    return quality_values


def quality_passed(item):
    quality = get_quality_values(item)
    for k, v in quality.items():
        if v != 'PASSED':
            return False
    return True


def filter_corrupted_items(items, workers=10, verbose=True):
    valid_flag = parallelize(quality_passed,
                             items,
                             max_workers=workers,
                             progressbar=False)
    if verbose:
        corrupted_items_ids = [item.id for item, flag in zip(items, valid_flag)
                               if not flag]
        nc = len(corrupted_items_ids)
        if nc:
            logger.warning(f"Discarding {nc} / {len(items)} corrupted "
                           f"products: {corrupted_items_ids}")

    items.items = [i for i, flag in zip(items, valid_flag) if flag]

    return items


def mask_clouds(darr, scl_mask):
    """darr has dims (time, band, y, x),
    mask has dims (time, band, y, x)"""
    if isinstance(darr.data, da.core.Array):
        mask = da.broadcast_to(scl_mask.data, darr.shape)
        darr_masked = da.where(~mask, 0, darr)
    else:
        mask = np.broadcast_to(scl_mask.data, darr.shape)
        darr_masked = darr.where(mask, 0)
    return darr.copy(data=darr_masked)


def force_unique_time(darr):
    """Add microseconds to time vars which repeats in order to make the
    time index of the DataArray unique, as sometimes observations from the same
    day can be split in multiple obs"""
    unique_ts, counts_ts = np.unique(darr.time, return_counts=True)
    double_ts = unique_ts[np.where(counts_ts > 1)]

    new_time = []
    c = 0
    for i in range(darr.time.size):
        v = darr.time[i].values
        if v in double_ts:
            v = v + c
            c += 1
        new_time.append(v)
    new_time = np.array(new_time)
    darr['time'] = new_time
    return darr


def harmonize_tmp(data):
    """
    Harmonize new Sentinel-2 data to the old baseline.

    https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a#Baseline-Change
    https://github.com/microsoft/PlanetaryComputer/issues/134

    Parameters
    ----------
    data: xarray.DataArray
        A DataArray with four dimensions: time, band, y, x

    Returns
    -------
    harmonized: xarray.DataArray
        A DataArray with all values harmonized to the old
        processing baseline.
    """
    baseline = data.coords['s2:processing_baseline'].astype(float)
    baseline_flag = baseline < 4

    if all(baseline_flag):
        return data

    offset = 1000
    bands = ["B01", "B02", "B03", "B04",
             "B05", "B06", "B07", "B08",
             "B8A", "B09", "B10", "B11", "B12"]

    old = data.isel(time=baseline_flag)
    to_process = list(set(bands) & set(data.band.data.tolist()))

    new = data.sel(time=~baseline_flag).drop_sel(band=to_process)

    new_harmonized = data.sel(time=~baseline_flag, band=to_process).copy()

    new_harmonized = new_harmonized.clip(offset)
    new_harmonized -= offset

    new = xr.concat([new, new_harmonized], "band").sel(
        band=data.band.data.tolist())
    return xr.concat([old, new], dim="time")


def harmonize(data):
    """
    Harmonize new Sentinel-2 data to the old baseline.

    Parameters
    ----------
    data: xarray.DataArray
        A DataArray with four dimensions: time, band, y, x

    Returns
    -------
    harmonized: xarray.DataArray
        A DataArray with all values harmonized to the old
        processing baseline.
    """
    cutoff = datetime.datetime(2022, 1, 25)
    offset = 1000
    bands = ["B01", "B02", "B03", "B04",
             "B05", "B06", "B07", "B08",
             "B8A", "B09", "B10", "B11", "B12"]

    old = data.sel(time=slice(cutoff))

    to_process = list(set(bands) & set(data.band.data.tolist()))
    new = data.sel(time=slice(cutoff, None)).drop_sel(band=to_process)

    new_harmonized = data.sel(time=slice(
        cutoff, None), band=to_process).clip(offset)
    new_harmonized -= offset

    new = xr.concat([new, new_harmonized], "band").sel(
        band=data.band.data.tolist())
    return xr.concat([old, new], dim="time")


def query_l2a_items(tile,
                    start_date,
                    end_date,
                    max_cloud_cover,
                    filter_corrupted):
    import pystac_client
    import planetary_computer
    from satio_pc.extension import ESAWorldCoverTimeSeries  # noqa register extension

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    time_range = f"{start_date}/{end_date}"

    query_params = {"eo:cloud_cover": {"lt": max_cloud_cover},
                    "s2:mgrs_tile": {"eq": tile}}

    search = catalog.search(collections=["sentinel-2-l2a"],
                            datetime=time_range,
                            query=query_params)
    items = search.item_collection()

    if filter_corrupted:
        items = filter_corrupted_items(items)

    return items


def load_l2a(bounds,
             epsg,
             tile,
             start_date,
             end_date,
             bands=None,
             max_cloud_cover=90,
             filter_corrupted=True):
    import stackstac

    items = query_l2a_items(tile,
                            start_date,
                            end_date,
                            max_cloud_cover,
                            filter_corrupted)

    assets_10m = ['B02', 'B03', 'B04', 'B08']
    assets_20m = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
    assets_60m = ['B01', 'B09']

    if bands is None:
        bands = assets_10m + assets_20m + assets_60m

    scl = 'SCL'

    ds = {}
    assets = {10: [b for b in bands if b in assets_10m],
              20: [b for b in bands if b in assets_20m],
              60: [b for b in bands if b in assets_60m],
              'scl': [scl]}

    chunksize = {10: 1024,
                 20: 512,
                 60: 512,
                 'scl': 512}

    dtype = {10: np.uint16,
             20: np.uint16,
             60: np.uint16,
             'scl': np.uint8}

    keep_vars = ['time', 'band', 'y', 'x', 'id', 's2:processing_baseline']
    for res in assets.keys():
        if len(assets[res]) == 0:
            continue
        ds[res] = stackstac.stack(items,
                                  assets=assets[res],
                                  epsg=f'EPSG:{epsg}',
                                  bounds=bounds,
                                  chunksize=chunksize[res],
                                  xy_coords='center',
                                  rescale=False,
                                  dtype=dtype[res],
                                  fill_value=0)
        del ds[res].attrs['spec']
        ds_vars = list(ds[res].coords.keys())
        drop_vars = [v for v in ds_vars if v not in keep_vars]
        ds[res] = ds[res].drop_vars(drop_vars)
        ds[res] = force_unique_time(ds[res])

        # coerce dtypes
        for v in ['id', 'band', 's2:processing_baseline']:
            ds[res][v] = ds[res][v].astype(str)

        if res in (10, 20, 60):
            if len(assets[res]) == 0:
                continue
            # harmonize values for processing baseline 4.0 (25th Jan 2022)
            ds[res] = ds[res].ewc.harmonize()

    return ds


def preprocess_l2a_cache(ds_dict,
                         clouds_mask,
                         start_date,
                         end_date,
                         composite_freq=10,
                         composite_window=20,
                         composite_mode='median',
                         reflectance=True,
                         tmpdir='.'):

    ds10_block = ds_dict[10]
    ds20_block = ds_dict[20]
    scl20_block = clouds_mask

    timer10 = FeaturesTimer(10, 'l2a')
    timer20 = FeaturesTimer(20, 'l2a')

    with tempfile.TemporaryDirectory(prefix='ewc_tmp-', dir=tmpdir) as \
            tmpdirname:

        # download
        logger.info("Loading block data")
        timer10.load.start()
        ds10_block = ds10_block.ewc.cache(tmpdirname)
        timer10.load.stop()

        timer20.load.start()
        ds20_block = ds20_block.ewc.cache(tmpdirname)
        scl20_block = scl20_block.ewc.cache(tmpdirname)
        scl10_block = scl20_block.ewc.rescale(scale=2,
                                              order=0)
        scl10_block = scl10_block.ewc.cache(tmpdirname)
        timer20.load.stop()

        # 10m
        # mask clouds
        timer10.composite.start()
        ds10_block_masked = ds10_block.ewc.mask(
            scl10_block).ewc.cache(tmpdirname)

        logger.info("Compositing 10m block data")
        # composite
        ds10_block_comp = ds10_block_masked.ewc.composite(
            freq=composite_freq,
            window=composite_window,
            start=start_date,
            end=end_date).ewc.cache(tmpdirname)
        timer10.composite.stop()

        logger.info("Interpolating 10m block data")
        # interpolation
        timer10.interpolate.start()
        ds10_block_interp = ds10_block_comp.ewc.interpolate(
        ).ewc.cache(tmpdirname)
        timer10.interpolate.stop()

        # 20m
        # mask
        timer20.composite.start()
        ds20_block_masked = ds20_block.ewc.mask(
            scl20_block).ewc.cache(tmpdirname)

        logger.info("Compositing 20m block data")
        # composite
        ds20_block_comp = ds20_block_masked.ewc.composite(
            freq=composite_freq,
            window=composite_window,
            start=start_date,
            end=end_date).ewc.cache(tmpdirname)
        timer20.composite.stop()

        logger.info("Interpolating 20m block data")
        # interpolation
        timer20.interpolate.start()
        ds20_block_interp = ds20_block_comp.ewc.interpolate(
        ).ewc.cache(tmpdirname)
        timer20.interpolate.stop()

        logger.info("Merging 10m and 20m series")
        # merging to 10m cleaned data
        ds20_block_interp_10m = ds20_block_interp.ewc.rescale(scale=2,
                                                              order=1,
                                                              nodata_value=0)
        dsm10 = xr.concat([ds10_block_interp,
                           ds20_block_interp_10m],
                          dim='band')

        if reflectance:
            dsm10 = dsm10.astype(np.float32) / 10000

        dsm10.attrs = ds10_block.attrs

        for t in timer10, timer20:
            t.load.log()
            t.composite.log()
            t.interpolate.log()

        for t in timer10, timer20:
            t.log()

    dsm10 = dsm10.ewc.cache(tmpdir)

    return dsm10


def preprocess_l2a(ds_dict,
                   clouds_mask,
                   start_date,
                   end_date,
                   composite_freq=10,
                   composite_window=20,
                   composite_mode='median',
                   reflectance=True):

    ds10_block = ds_dict[10]
    ds20_block = ds_dict[20]
    scl20_block = clouds_mask

    timer10 = FeaturesTimer(10, 'l2a')
    timer20 = FeaturesTimer(20, 'l2a')

    # download
    logger.info("Loading block data")
    timer10.load.start()
    ds10_block = ds10_block.ewc.persist_chunk()
    timer10.load.stop()

    timer20.load.start()
    ds20_block = ds20_block.ewc.persist_chunk()
    scl20_block = scl20_block.ewc.persist_chunk()
    scl10_block = scl20_block.ewc.rescale(scale=2,
                                          order=0)
    scl10_block = scl10_block.ewc.persist_chunk()
    timer20.load.stop()

    # 10m
    # mask clouds
    timer10.composite.start()
    ds10_block_masked = ds10_block.ewc.mask(
        scl10_block).ewc.persist_chunk()

    logger.info("Compositing 10m block data")
    # composite
    ds10_block_comp = ds10_block_masked.ewc.composite(
        freq=composite_freq,
        window=composite_window,
        mode=composite_mode,
        start=start_date,
        end=end_date).ewc.persist_chunk()
    timer10.composite.stop()

    logger.info("Interpolating 10m block data")
    # interpolation
    timer10.interpolate.start()
    ds10_block_interp = ds10_block_comp.ewc.interpolate(
    ).ewc.persist_chunk()
    timer10.interpolate.stop()

    # 20m
    # mask
    timer20.composite.start()
    ds20_block_masked = ds20_block.ewc.mask(
        scl20_block).ewc.persist_chunk()

    logger.info("Compositing 20m block data")
    # composite
    ds20_block_comp = ds20_block_masked.ewc.composite(
        freq=composite_freq,
        window=composite_window,
        mode=composite_mode,
        start=start_date,
        end=end_date).ewc.persist_chunk()
    timer20.composite.stop()

    logger.info("Interpolating 20m block data")
    # interpolation
    timer20.interpolate.start()
    ds20_block_interp = ds20_block_comp.ewc.interpolate(
    ).ewc.persist_chunk()
    timer20.interpolate.stop()

    logger.info("Merging 10m and 20m series")
    # merging to 10m cleaned data
    ds20_block_interp_10m = ds20_block_interp.ewc.rescale(scale=2,
                                                          order=1,
                                                          nodata_value=0)
    dsm10 = xr.concat([ds10_block_interp,
                       ds20_block_interp_10m],
                      dim='band')

    if reflectance:
        dsm10 = dsm10.astype(np.float32) / 10000

    dsm10.attrs = ds10_block.attrs

    for t in timer10, timer20:
        t.load.log()
        t.composite.log()
        t.interpolate.log()

    for t in timer10, timer20:
        t.log()

    dsm10 = dsm10.ewc.persist_chunk()

    return dsm10
