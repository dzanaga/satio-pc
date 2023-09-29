import shutil
from pathlib import Path
from loguru import logger

from satio_pc.utils import random_string
from satio_pc.utils.azure import AzureBlobReader


DEFAULT_SETTINGS = {

    "l2a": {
        "max_cloud_cover": 90,
        "composite": {"freq": 10,
                      "window": 20,
                      "mode": "median"},
        "mask": {"erode_r": 3,
                 "dilate_r": 13,
                 "max_invalid_ratio": 1},
        "bands": ['B02', 'B03', 'B04', 'B08', 'B11', 'B12'],
        "indices": ["ndvi"],
        "percentiles": [10, 25, 50, 75, 90],
    },

    "gamma0": {
        "composite": {"freq": 10,
                      "window": 10,
                      "mode": "median"}},
}

# parser = argparse.ArgumentParser()
# parser.add_argument('tile')
# parser.add_argument('block_id', type=int)
# parser.add_argument('year', type=int)
# parser.add_argument('-c', '--cleanup', action='store_true')
# args = parser.parse_args()


class S2BlockExtractor:

    def __init__(self,
                 tile,
                 block_id,
                 year,
                 settings=None,
                 output_folder='.',
                 connection_str=None,
                 container_name=None,
                 cleanup=True,
                 terminate_if_failed=False,) -> None:

        self.tile = tile
        self.block_id = block_id
        self.year = year

        self.output_folder = Path(output_folder)

        self.sensor = 's2'
        self._cleanup = cleanup
        self._terminate_if_failed = terminate_if_failed

        self.block_folder = Path(output_folder) / f'ewc_{tile}_{block_id:03d}'
        self.block_folder.mkdir(exist_ok=True, parents=True)

        if (connection_str is not None) and (container_name is not None):
            self._azure_client = AzureBlobReader(
                connection_str, container_name)
        else:
            self._azure_client = None

        self.local_log = {
            k: self.block_folder /
            f'{k}_{self.tile}_{self.block_id:03d}_{self.year}.log'
            for k in ('done', 'error', 'proc')}

        self.azure_log = {
            k: f"logs/{k}/{self.year}/{self.sensor}/{self.local_log[k].name}"
            for k in ('done', 'error', 'proc')}

        self._settings = settings or DEFAULT_SETTINGS
        self._bands = self._settings['l2a']['bands']
        self._indices = self._settings['l2a']['indices']
        self._percentiles = self._settings['l2a']['percentiles']

    def upload_results(self, fn):

        if self._azure_client is None:
            return None
        else:
            azure: AzureBlobReader = self._azure_client

        if fn is None:
            azure.upload_file(self.local_log['error'],
                              self.azure_log['error'],
                              overwrite=True)
        else:
            # upload to azure
            dst_fn = (f"features/{self.year}/{self.sensor}/{self.tile}/"
                      f"{Path(fn).name}")
            logger.info(f"Uploading features to {dst_fn}")
            azure.upload_file(fn,
                              dst_fn,
                              overwrite=True)

            s = logger.add(self.local_log['done'])
            logger.success(f"{dst_fn} uploaded")
            logger.remove(s)

            azure.upload_file(self.local_log['done'],
                              self.azure_log['done'],
                              overwrite=True)

        azure.upload_file(self.local_log['proc'],
                          self.azure_log['proc'],
                          overwrite=True)

    def extract(self, overwrite=False):

        done = self._azure_client.check_file_exists(self.azure_log['done'])
        if done and not overwrite:
            logger.warning(f"Block {self.tile} {self.block_id} already "
                           "processed, skipping.")
            return None
        else:
            for v in self.azure_log.values():
                self._azure_client.delete_file(v)

        logger.add(self.local_log['proc'])

        # get features to tif
        fn = self._extract_s2_wrapper()
        self.upload_results(fn)

        logger.info("Cleaning up...")
        if self._cleanup:
            shutil.rmtree(self.block_folder)
        logger.success("Done")

        if fn is None and self._terminate_if_failed:
            import sys
            sys.exit(5)

        return None

    def _save_features(self, data, fn, bounds, epsg):
        logger.info(f"Saving features stack to {fn}")
        data.ewc.save_features(fn, bounds, epsg)

    def _extract_s2_wrapper(self):
        try:
            data, fn, bounds, epsg = self._extract_s2()
            self._save_features(data, fn, bounds, epsg)
            return fn

        except Exception as e:
            s = logger.add(self.local_log['error'])
            logger.exception(f"Features extraction failed: {e}")
            logger.remove(s)
            return None

    def _extract_s2(self):
        import xarray as xr
        from loguru import logger

        from satio_pc.sentinel2 import load_l2a, preprocess_l2a
        from satio_pc.preprocessing.clouds import preprocess_scl
        from satio_pc.grid import get_blocks_gdf

        year = self.year
        tile = self.tile
        block_id = self.block_id

        start_date = f'{year}-01-01'
        end_date = f'{year + 1}-01-01'
        max_cloud_cover = self._settings['l2a']['max_cloud_cover']

        blocks = get_blocks_gdf([tile])
        block = blocks[blocks.block_id == block_id].iloc[0]

        s2_dict = load_l2a(block.bounds,
                           block.epsg,
                           block.tile,
                           start_date,
                           end_date,
                           bands=self._bands,
                           max_cloud_cover=max_cloud_cover)

        # mask preparation
        mask_settings = self._settings['l2a']['mask']
        scl = preprocess_scl(s2_dict['scl'],
                             **mask_settings)

        scl20_mask = scl.mask
        scl20_aux = scl.aux

        s2 = preprocess_l2a(s2_dict,
                            scl20_mask,
                            start_date,
                            end_date,
                            composite_freq=self._settings[
                                'l2a']['composite']['freq'],
                            composite_window=self._settings[
                                'l2a']['composite']['window'],
                            composite_mode=self._settings[
                                'l2a']['composite']['mode'])

        s2_indices = self._indices

        # compute indices
        s2_vi = s2.ewc.indices(s2_indices)

        # percentiles sensors and vis
        q = self._percentiles
        ps = [s.ewc.percentile(q, name_prefix='s2') for s in (s2, s2_vi)]

        # fix time to same timestamp (only 1) to avoid concat issues
        # (different compositing settings for s2 and s1)
        for p in ps:
            p['time'] = ps[0].time

        # scl aux 10m
        scl10_aux = scl20_aux.ewc.rescale(scale=2, order=1)
        scl10_aux['time'] = ps[0].time

        final = xr.concat(ps + [scl10_aux], dim='band')
        final.name = 'satio-features-s2'

        logger.info("Computing features stack")
        final = final.persist()
        final = final.squeeze()

        output_folder = Path(self.block_folder)
        fn = output_folder / \
            f'{final.name}_{tile}_{block.block_id:03d}_{year}.tif'

        return final, fn, block.bounds, block.epsg


class S2BlockExtractorHabitat(S2BlockExtractor):

    def _extract_s2(self):
        import xarray as xr
        import dask.array as da
        from pyproj.crs import CRS
        from loguru import logger
        import tempfile
        from satio_pc.sentinel2 import load_l2a, preprocess_l2a
        from satio_pc.preprocessing.clouds import preprocess_scl
        from satio_pc._habitat import RSI_META_S2_HABITAT
        from satio_pc.grid import get_blocks_gdf, tile_to_epsg

        year = self.year
        tile = self.tile
        block_id = self.block_id

        start_date = f'{year}-01-01'
        end_date = f'{year + 1}-01-01'
        max_cloud_cover = self._settings['l2a']['max_cloud_cover']

        blocks = get_blocks_gdf([tile])
        block = blocks[blocks.block_id == block_id].iloc[0]

        s2_dict = load_l2a(block.bounds,
                           block.epsg,
                           block.tile,
                           start_date,
                           end_date,
                           max_cloud_cover=max_cloud_cover)

        # preprocess s2
        tmpdir = tempfile.TemporaryDirectory(
            prefix='ewc_tmp-', dir=self.block_folder)

        # mask preparation
        mask_settings = self._settings['l2a']['mask']
        scl = preprocess_scl(s2_dict['scl'],
                             **mask_settings)

        scl20_mask = scl.mask
        scl20_aux = scl.aux

        s2 = preprocess_l2a(s2_dict,
                            scl20_mask,
                            start_date,
                            end_date,
                            composite_freq=self._settings['l2a']['composite']['freq'],
                            composite_window=self._settings['l2a']['composite'][
                                'window'],
                            tmpdir=tmpdir.name)

        s2_indices = list(RSI_META_S2_HABITAT.keys())

        # compute indices
        s2_vi = s2.ewc.indices(s2_indices,
                               rsi_meta=RSI_META_S2_HABITAT)

        # percentiles sensors and vis
        q = [10, 25, 50, 75, 90]
        ps = [s.ewc.percentile(q, name_prefix='s2') for s in (s2, s2_vi)]

        # fix time to same timestamp (only 1) to avoid concat issues
        # (different compositing settings for s2 and s1)
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
                                       'band': [f's2-ndvi-ts{i}'
                                                for i in range(1, 13)],
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

        output_folder = Path(self.block_folder)
        fn = output_folder / \
            f'{final.name}_{tile}_{block.block_id:03d}_{year}.tif'
        logger.info(f"Saving features stack to {fn}")
        final_ds.rio.to_raster(fn,
                               windowed=False,
                               tiled=True,
                               compress='deflate',
                               predictor=3,
                               zlevel=4)

        return fn


class S2Extractor:

    def __init__(self,
                 settings=None) -> None:

        self.sensor = 's2'

        self._settings = settings or DEFAULT_SETTINGS['l2a']
        self._bands = self._settings['bands']
        self._indices = self._settings['indices']
        self._percentiles = self._settings['percentiles']

    def extract(self, year, tile, bounds, epsg):
        import xarray as xr
        from loguru import logger

        from satio_pc.sentinel2 import load_l2a, preprocess_l2a
        from satio_pc.preprocessing.clouds import preprocess_scl

        start_date = f'{year}-01-01'
        end_date = f'{year + 1}-01-01'
        max_cloud_cover = self._settings['max_cloud_cover']

        s2_dict = load_l2a(bounds,
                           epsg,
                           tile,
                           start_date,
                           end_date,
                           bands=self._bands,
                           max_cloud_cover=max_cloud_cover)

        # mask preparation
        mask_settings = self._settings['mask']
        scl = preprocess_scl(s2_dict['scl'],
                             **mask_settings)

        scl20_mask = scl.mask
        scl20_aux = scl.aux

        s2 = preprocess_l2a(s2_dict,
                            scl20_mask,
                            start_date,
                            end_date,
                            composite_freq=self._settings[
                                'composite']['freq'],
                            composite_window=self._settings[
                                'composite']['window'],
                            composite_mode=self._settings[
                                'composite']['mode'])

        s2_indices = self._indices

        # compute indices
        s2_vi = s2.ewc.indices(s2_indices)

        # percentiles sensors and vis
        q = self._percentiles
        ps = [s.ewc.percentile(q, name_prefix='s2') for s in (s2, s2_vi)]

        # fix time to same timestamp (only 1) to avoid concat issues
        # (different compositing settings for s2 and s1)
        for p in ps:
            p['time'] = ps[0].time

        # scl aux 10m
        scl10_aux = scl20_aux.ewc.rescale(scale=2, order=1)
        scl10_aux['time'] = ps[0].time

        final = xr.concat(ps + [scl10_aux], dim='band')
        final.name = 'satio-features-s2'

        logger.info("Computing features stack")
        final = final.persist()
        final = final.squeeze()

        return final
