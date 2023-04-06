import argparse
import shutil
from pathlib import Path
from loguru import logger


connect_str = "DefaultEndpointsProtocol=https;AccountName=planetarycomputervito;AccountKey=k8ai82LF2s3RTenyQHGQW7WPMrN4+ly69oYY0mU5ATaUSLT6LJiRWNrBUQc09oTh0uJvKao6Kj5x+AStSLOTZg==;EndpointSuffix=core.windows.net"

settings = {
    
    "l2a": {
        "max_cloud_cover": 90,
        "composite": {"freq": 10, "window": 20},
        "mask": {"erode_r": 3,
                 "dilate_r": 13,
                 "max_invalid_ratio": 1}},
    
    "gamma0" : {
        "composite": {"freq": 10, "window": 10}},
}
  

def upload_blob(fn, dst_fn, container_name):
    import azure.storage
    from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
    
    fn = str(fn)
    dst_fn = str(dst_fn)
    
    
    # Create a BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # Create a ContainerClient object
    try:
        container_client = blob_service_client.get_container_client(container_name)
    except:
        container_client = blob_service_client.create_container(container_name)

    # Upload the file to Blob Storage
    with open(fn, 'rb') as data:
        blob_client = container_client.upload_blob(name=dst_fn, data=data)


def _extract_s2_features(tile, block_id, year, output_folder='.'):
    import xarray as xr
    import dask.array as da
    from pyproj.crs import CRS
    from dask import delayed
    from loguru import logger
    import numpy as np
    import tempfile
    import geopandas as gpd
    from satio_pc.sentinel2 import load_l2a, preprocess_l2a
    from satio_pc.preprocessing.clouds import preprocess_scl
    from satio_pc.sentinel1 import load_gamma0, preprocess_gamma0
    from satio_pc._habitat import RSI_META_S2_HABITAT
    from satio_pc.grid import get_blocks_gdf, tile_to_epsg
    
    start_date = f'{year}-01-01'
    end_date = f'{year + 1}-01-01'
    max_cloud_cover = settings['l2a']['max_cloud_cover']
    
    blocks = get_blocks_gdf([tile])
    block = blocks[blocks.block_id == block_id].iloc[0]
    
    s2_dict = load_l2a(block.bounds,
                       block.epsg,
                       block.tile,
                       start_date,
                       end_date,
                       max_cloud_cover=max_cloud_cover)
    
    # preprocess s2
    tmpdir = tempfile.TemporaryDirectory(prefix='ewc_tmp-', dir='/tmp')

    # mask preparation
    mask_settings = settings['l2a']['mask']
    scl = preprocess_scl(s2_dict['scl'],
                         **mask_settings)

    scl20_mask = scl.mask
    scl20_aux = scl.aux

    s2 = preprocess_l2a(s2_dict,
                        scl20_mask,
                        start_date,
                        end_date,
                        composite_freq=settings['l2a']['composite']['freq'], 
                        composite_window=settings['l2a']['composite']['window'],
                        tmpdir=tmpdir.name)

    s2_indices = list(RSI_META_S2_HABITAT.keys())

    # compute indices
    s2_vi = s2.ewc.indices(s2_indices,
                           rsi_meta=RSI_META_S2_HABITAT)

    # percentiles sensors and vis
    q = [10, 25, 50, 75, 90]
    ps = [s.ewc.percentile(q, name_prefix='s2') for s in (s2, s2_vi)]

    # fix time to same timestamp (only 1) to avoid concat issues (different compositing settings for s2 and s1)
    for p in ps:
        p['time'] = ps[0].time

    # ndvi 12 timestamps
    ndvi_ts = s2_vi.sel(band=['ndvi'])
    ndvi_ts = ndvi_ts.ewc.composite(freq=30,
                                    window=30,
                                    start=start_date,
                                    end=end_date)

    ndvi_ts = xr.DataArray(da.transpose(ndvi_ts.data, (1, 0, 2, 3)),
                                 dims=ps[0].dims,
                                 coords={'time': ps[0].time,
                                         'band': [f's2-ndvi-ts{i}' for i in range(1, 13)],
                                         'y': ps[0].y,
                                         'x': ps[0].x},
                                 attrs=ps[0].attrs)

    # scl aux 10m
    scl10_aux = scl20_aux.ewc.rescale(scale=2, order=1)
    scl10_aux['time'] = ps[0].time                                           

    final = xr.concat(ps + [ndvi_ts, scl10_aux], dim='band')
    final.name = 'satio-features-s2'
    
    logger.info("Computing features stack")
    final = final.persist()
    final = final.squeeze()
    
    epsg = tile_to_epsg(tile)
    crs = CRS.from_epsg(epsg)
    final = final.rio.write_crs(crs)
    final_ds = final.to_dataset('band')
    
    output_folder = Path(output_folder)
    fn = output_folder / f'{final.name}_{tile}_{block.block_id:03d}_{year}.tif'
    logger.info(f"Saving features stack to {fn}")
    final_ds.rio.to_raster(fn,
                           windowed=False,
                           tiled=True,
                           compress='deflate',
                           predictor=3,
                           zlevel=4)
    
    return fn
    

def extract_s2_features(tile, block_id, year, output_folder='.'):
    output_folder = Path(output_folder)
    
    try:
        fn = _extract_s2_features(tile, block_id, year, output_folder)
        return fn
    except Exception as e:
        log_fn = output_folder / f'ERROR_{tile}_{block_id:03d}.log'
        s = logger.add(log_fn)
        logger.exception(f"Features extraction failed: {e}")
        logger.remove(s)
        return None
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('tile')
    parser.add_argument('block_id', type=int)
    parser.add_argument('year', type=int)
    parser.add_argument('-c', '--cleanup', action='store_true')
    
    args = parser.parse_args()
    
    cleanup = args.cleanup
    container_name = 'habitattest'
    sensor = 's2'
    
    tile = args.tile
    block_id = args.block_id
    year = args.year
    
    output_folder = Path(f'ewc_{tile}_{block_id:03d}')
    output_folder.mkdir(exist_ok=True, parents=True)
    
    log_fn = output_folder / f'PROC_{tile}_{block_id:03d}.log'
    logger.add(log_fn)
    
    # get features to tif
    fn = extract_s2_features(tile, block_id, year, output_folder)
    
    if fn is None:
        err_log = output_folder / f'ERROR_{tile}_{block_id:03d}.log'
        upload_blob(err_log, f"logs/errors/{err_log}", container_name)
    
    else:
        # upload to azure
        dst_fn = f"features/{tile}/{year}/{sensor}/{Path(fn).name}"
        logger.info(f"Uploading features to {dst_fn}")
        upload_blob(str(fn), dst_fn, container_name)
        logger.success(f"{dst_fn} uploaded")

    upload_blob(log_fn, f"logs/proc/{log_fn}", container_name)
    
    logger.info("Cleaning up...")
    if cleanup:
        shutil.rmtree(output_folder)
    logger.success("Done")
    
    if fn is None:
        import sys
        sys.exit(5)