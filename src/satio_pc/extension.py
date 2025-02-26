import atexit
import tempfile
import warnings
from typing import Dict

import dask.array as da
import numpy as np
import xarray as xr
from tqdm import tqdm

from satio_pc import random_string
from satio_pc.features import percentile
from satio_pc.indices import rsi_ts
from satio_pc.preprocessing import get_yx_coords, to_dataarray_coords
from satio_pc.preprocessing.composite import calculate_moving_composite
from satio_pc.preprocessing.interpolate import interpolate_ts_linear
from satio_pc.preprocessing.rescale import rescale_ts
from satio_pc.preprocessing.speckle import multitemporal_speckle_ts
from satio_pc.sentinel2 import harmonize, mask_clouds
from satio_pc.superres import (
    MODELS_NAMES_CV,
    MODELS_NAMES_SUPERIMAGE,
)

SUPPORTED_SUPERRES_MODELS = list(MODELS_NAMES_CV.keys()) + list(
    MODELS_NAMES_SUPERIMAGE.keys()
)


@xr.register_dataarray_accessor("satio")
class SatioTimeSeries:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._obj.attrs["bounds"] = self.bounds
        # run check that we have a timeseries
        # assert xarray_obj.dims == ('time', 'band', 'y', 'x')
        self._superres_models = {}

    def rescale(
        self,
        scale=2,
        order=1,
        preserve_range=True,
        anti_aliasing=None,
        anti_aliasing_sigma=None,
        nodata_value=None,
    ):
        return rescale_ts(
            self._obj,
            scale=scale,
            order=order,
            preserve_range=preserve_range,
            anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma,
            nodata_value=nodata_value,
        )

    def _load_superres_model(self, model_name):
        from satio_pc.superres import (
            SuperImage,
            SuperResCV,
        )

        model = self._superres_models.get(model_name)
        if model is None:
            if model_name in MODELS_NAMES_CV.keys():
                model = SuperResCV(model_name)
            elif model_name in MODELS_NAMES_SUPERIMAGE.keys():
                model = SuperImage(model_name)
            else:
                raise ValueError(
                    f"Invalid model name: {model_name}. "
                    "Should be one of "
                    f"{SUPPORTED_SUPERRES_MODELS}."
                )
            self._superres_models[model_name] = model
        return model

    def superscale(self, scale=4, model_name="edsr-base", progress_bar=True):
        """
        Upscale an image by a given scale factor using a specified model.

        Args:
            scale (int): The scale factor to use for the upscaling.
                Supported values are 2, 3, 4. (For 'lapsrn' only (2, 4, 8).
                Default is 4.
            model_name (str): The name of the model to use for the
                upscaling. Should be one of ('lapsrn', 'espcn', 'fsrcnn',
                'drln', 'drln-bam', 'mdsr', 'edsr-base', 'edsr', 'msrn',
                'a2n', 'pan').
                Default is 'edsr-base'.

        Returns:
            xr.DataArray: The upscaled timeseries.
        """
        model = self._load_superres_model(model_name)

        data = self._obj.data

        t, b, y, x = data.shape
        # data = np.reshape(data, (t * b, y, x))

        range_t = tqdm(range(t)) if progress_bar else range(t)
        new_data = [
            model.upscale(data[ti], progress_bar=False, scale=scale)
            for ti in range_t
        ]
        new_data = np.array(new_data)
        new_data = new_data.astype(data.dtype)
        # _, new_y, new_x = new_data.shape
        # new_data = np.reshape(new_data, (t, b, new_y, new_x))

        new_coords = self._obj.coords.to_dataset().drop_vars(["x", "y"])
        y, x = get_yx_coords(new_data, self.bounds)
        new_coords["y"] = y
        new_coords["x"] = x

        # epsg = self._obj.attrs.get('epsg', None)
        # new_obj = to_dataarray(new_data,
        #                        self.bounds,
        #                        epsg=epsg,
        #                        bands=self._obj.band,
        #                        time=self._obj.time,
        #                        attrs=self._obj.attrs)
        new_obj = to_dataarray_coords(new_data, self._obj.dims, new_coords)
        return new_obj

    def mask(self, mask, nodata_value=0):
        return mask_clouds(self._obj, mask, nodata_value)

    def preprocess_scl(
        self,
        erode_r=3,
        dilate_r=13,
        max_invalid_ratio=1,
        snow_dilate_r=3,
        max_invalid_snow_cover=0.9,
    ):
        """
        Preprocess Sentinel-2 Scene Classification Map (SCL) to remove clouds
        and snow pixels.

        Args:
            erode_r (int): Radius of the erosion kernel used to remove small
                cloud pixels.
            dilate_r (int): Radius of the dilation kernel used to increase
                clouds footprints after erosion.
            max_invalid_ratio (float): Maximum ratio of invalid pixels (clouds,
                cloud shadows, snow) allowed in the SCL. If the ratio is above
                this threshold, the pixels are not masked. Default is 1.
            snow_dilate_r (int): Radius of the dilation kernel used to remove
                snow pixels.
            max_invalid_snow_cover (float): Maximum ratio of snow pixels
                in the time series. If the ratio is above this threshold,
                the pixels are not masked. Default is 0.9.

        Returns:
            tuple: A tuple containing the mask and auxiliary data
            xr.DataArrays.
        """
        if "SCL" not in self._obj.band.values:
            raise ValueError("SCL band not found in the time series.")

        from satio_pc.preprocessing.clouds import preprocess_scl

        scl = preprocess_scl(
            self._obj.sel(band=["SCL"]),
            erode_r=erode_r,
            dilate_r=dilate_r,
            max_invalid_ratio=max_invalid_ratio,
            snow_dilate_r=snow_dilate_r,
            max_invalid_snow_cover=max_invalid_snow_cover,
        )
        return scl.mask, scl.aux

    def composite(
        self,
        freq=7,
        window=None,
        start=None,
        end=None,
        mode="median",
        use_all_obs=True,
    ):
        return calculate_moving_composite(
            self._obj, freq, window, start, end, mode, use_all_obs
        )

    def interpolate(self):
        if isinstance(self._obj.data, da.core.Array):
            darr_interp = da.map_blocks(
                interpolate_ts_linear,
                self._obj.data,
                dtype=self._obj.dtype,
                chunks=self._obj.chunks,
            )
        else:
            darr_interp = interpolate_ts_linear(self._obj.data)

        out = self._obj.copy(data=darr_interp)
        return out

    def multitemporal_speckle(self, kernel="gamma", mtwin=15, enl=7):
        return multitemporal_speckle_ts(self._obj, kernel, mtwin, enl)

    def indices(self, indices, clip=True, rsi_meta=None):
        """Compute Sentinel-2 / Sentinel-1 remote sensing indices"""
        return rsi_ts(self._obj, indices, clip, rsi_meta=rsi_meta)

    def percentile(self, q=[10, 25, 50, 75, 90], name_prefix="s2"):
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

    def coregister(
        self,
        reference_band: str = "B08",
        max_translation: int = 3,
        mask_zeros: bool = True,
    ):
        """
        Coregister the time-series data using the gradient of the median
        of a reference band.

        Args:
            reference_band (str): The reference band to use for coregistration.
            max_translation (int): The maximum number of pixels to translate
                the data.
            mask_zeros (bool): Whether to mask out zero values for the
                registration process. It is reccomened to set this to True and
                mask clouds before coregistering.

        Returns:
            The coregistered data.
        """
        from satio_pc.preprocessing.coregistration import coregister

        return coregister(
            self._obj,
            reference_band=reference_band,
            max_translation=max_translation,
            mask_zeros=mask_zeros,
        )

    def cache(self, tempdir=".", chunks=(-1, -1, 256, 256)):
        tmpfile = tempfile.NamedTemporaryFile(
            suffix=".nc", prefix="satio-", dir=tempdir
        )

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

    def save_features(
        self,
        filename,
        bounds,
        epsg,
        tags: Dict = None,
        compress_tag: str = "deflate-uint16",
        **profile_kwargs,
    ):
        from satio_pc.geotiff import save_features_geotiff

        # bounds = self.bounds
        # to be standardized as self._obj.epsg
        # epsg = int(self._obj.crs.split(':')[-1])
        bands_names = self._obj.band.values.tolist()

        if isinstance(self._obj.data, da.core.Array):
            data = self._obj.compute().data
        else:
            data = self._obj.data

        _ = save_features_geotiff(
            data,
            bounds=bounds,
            epsg=epsg,
            bands_names=bands_names,
            filename=filename,
            tags=tags,
            compress_tag=compress_tag,
            **profile_kwargs,
        )

    def add_band(self, data, name):
        data = xr.DataArray(
            np.expand_dims(data, axis=(0, 1)),
            dims=["time", "band", "y", "x"],
            coords={
                "time": self._obj.time,
                "band": [name],
                "y": self._obj.y,
                "x": self._obj.x,
            },
        )
        return xr.concat([self._obj, data], dim="band")

    def rgb(self, bands=None, vmin=0, vmax=1000, **kwargs):
        import hvplot.xarray  # noqa
        import hvplot.pandas  # noqa
        import panel as pn  # noqa
        import panel.widgets as pnw

        if self._obj.name is None:
            self._obj.name = f"satio-ts-{random_string(6)}"

        bands = ["B04", "B03", "B02"] if bands is None else bands
        im = self._obj.sel(band=bands).clip(vmin, vmax) / (vmax - vmin)
        return im.interactive.sel(time=pnw.DiscreteSlider).hvplot.rgb(
            x="x",
            y="y",
            bands="band",
            data_aspect=1,
            xaxis=None,
            yaxis=None,
            **kwargs,
        )

    def show(
        self, band=None, vmin=None, vmax=None, colormap="plasma", **kwargs
    ):
        import hvplot.xarray  # noqa
        import hvplot.pandas  # noqa
        import panel as pn  # noqa
        import panel.widgets as pnw

        if self._obj.name is None:
            self._obj.name = f"satio-ts-{random_string(6)}"

        im = self._obj
        band = im.band[0] if band is None else band
        im = im.sel(band=band)
        return im.interactive.sel(time=pnw.DiscreteSlider).hvplot(
            clim=(vmin, vmax),
            colormap=colormap,
            aspect=1,
            x="x",
            y="y",
            **kwargs,
        )
