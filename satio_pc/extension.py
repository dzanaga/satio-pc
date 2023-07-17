import warnings
import atexit
import tempfile
import xarray as xr
import dask.array as da
from typing import List, Dict

from satio_pc.preprocessing.composite import calculate_moving_composite
from satio_pc.preprocessing.interpolate import interpolate_ts_linear
from satio_pc.preprocessing.rescale import rescale_ts
from satio_pc.preprocessing.speckle import multitemporal_speckle_ts
from satio_pc.indices import rsi_ts
from satio_pc.features import percentile
from satio_pc.sentinel2 import mask_clouds, harmonize


@xr.register_dataarray_accessor("ewc")
class ESAWorldCoverTimeSeries:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._obj.attrs['bounds'] = self.bounds
        # run check that we have a timeseries
        # assert xarray_obj.dims == ('time', 'band', 'y', 'x')

    def rescale(self,
                scale=2,
                order=1,
                preserve_range=True,
                nodata_value=0):
        return rescale_ts(self._obj,
                          scale=scale,
                          order=order,
                          preserve_range=preserve_range,
                          nodata_value=nodata_value)

    def mask(self, mask):
        return mask_clouds(self._obj, mask)

    def composite(self,
                  freq=7,
                  window=None,
                  start=None,
                  end=None,
                  mode='median',
                  use_all_obs=True):
        return calculate_moving_composite(self._obj,
                                          freq,
                                          window,
                                          start,
                                          end,
                                          mode,
                                          use_all_obs)

    def interpolate(self):
        if isinstance(self._obj.data, da.core.Array):
            darr_interp = da.map_blocks(
                interpolate_ts_linear,
                self._obj.data,
                dtype=self._obj.dtype,
                chunks=self._obj.chunks)
        else:
            darr_interp = interpolate_ts_linear(self._obj.data)

        out = self._obj.copy(data=darr_interp)
        return out

    def multitemporal_speckle(self, kernel='gamma', mtwin=15, enl=7):
        return multitemporal_speckle_ts(self._obj, kernel, mtwin, enl)

    def indices(self, indices, clip=True, rsi_meta=None):
        """Compute Sentinel-2 / Sentinel-1 remote sensing indices"""
        return rsi_ts(self._obj, indices, clip, rsi_meta=rsi_meta)

    def percentile(self, q=[10, 25, 50, 75, 90], name_prefix='s2'):
        """Compute set of percentiles for the time-series bands"""
        return percentile(self._obj, q, name_prefix=name_prefix)

    @property
    def bounds(self):

        darr = self._obj

        res = darr.x[1] - darr.x[0]
        hres = res / 2

        xmin = (darr.x[0] - hres).values.tolist()
        xmax = (darr.x[-1] + hres).values.tolist()

        ymin = (darr.y[-1] - hres).values.tolist()
        ymax = (darr.y[0] + hres).values.tolist()

        return xmin, ymin, xmax, ymax

    def harmonize(self):
        return harmonize(self._obj)

    def cache(self, tempdir='.', chunks=(-1, -1, 256, 256)):
        tmpfile = tempfile.NamedTemporaryFile(suffix='.nc',
                                              prefix='satio-',
                                              dir=tempdir)

        chunks = self._obj.chunks if chunks is None else chunks

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._obj.to_netcdf(tmpfile.name)
            darr = xr.open_dataarray(tmpfile.name).chunk(chunks)

        atexit.register(tmpfile.close)
        return darr

    def persist_chunk(self, chunks=(-1, -1, 256, 256)):
        chunks = self._obj.chunks if chunks is None else chunks
        darr = self._obj.persist().chunk(chunks)
        return darr

    def save_features(self,
                      filename,
                      bounds,
                      epsg,
                      tags: Dict = None,
                      compress_tag: str = 'deflate-uint16',
                      **profile_kwargs):

        from satio_pc.geotiff import save_features_geotiff
        # bounds = self.bounds
        # to be standardized as self._obj.epsg
        # epsg = int(self._obj.crs.split(':')[-1])
        bands_names = self._obj.band.values.tolist()

        if isinstance(self._obj.data, da.core.Array):
            data = self._obj.compute().data
        else:
            data = self._obj.data

        _ = save_features_geotiff(data,
                                  bounds=bounds,
                                  epsg=epsg,
                                  bands_names=bands_names,
                                  filename=filename,
                                  tags=tags,
                                  compress_tag=compress_tag,
                                  **profile_kwargs)

    def rgb(self, bands=None, vmin=0, vmax=1000, **kwargs):
        import hvplot.xarray  # noqa
        import hvplot.pandas  # noqa
        import panel as pn  # noqa
        import panel.widgets as pnw

        bands = ['B04', 'B03', 'B02'] if bands is None else bands
        im = self._obj.sel(band=bands).clip(vmin, vmax) / (vmax - vmin)
        return im.interactive.sel(
            time=pnw.DiscreteSlider).hvplot.rgb(
                x='x', y='y',
                bands='band',
                data_aspect=1,
                xaxis=None,
                yaxis=None,
                **kwargs)

    def show(self, band=None, vmin=None, vmax=None,
             colormap='plasma', **kwargs):
        import hvplot.xarray  # noqa
        import hvplot.pandas  # noqa
        import panel as pn  # noqa
        import panel.widgets as pnw

        im = self._obj
        band = im.band[0] if band is None else band
        im = im.sel(band=band)
        return im.interactive.sel(time=pnw.DiscreteSlider).hvplot(
            clim=(vmin, vmax),
            colormap=colormap,
            aspect=1,
            x='x',
            y='y',
            **kwargs)
