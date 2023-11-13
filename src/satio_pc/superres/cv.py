import os
from pathlib import Path

import numpy as np


_DZ_WEIGHTS = Path('/data/users/Public/dzanaga/models_weights/opencv_superres_models/')  # noqa E501
_WEIGHT_BASE_URL = "https://test.com/"  # TODO: for download from bucket

MODELS_NAMES_CV = {'lapsrn': 'LapSRN',
                   'espcn': 'ESPCN',
                   'fsrcnn': 'FSRCNN'
                   }

_MODELS_SCALES = {'lapsrn': [2, 4, 8],
                  'espcn': [2, 3, 4],
                  'fsrcnn': [2, 3, 4]
                  }

_MODELS = MODELS_NAMES_CV.keys()


class SuperResCV:

    def __init__(self, model_name='lapsrn', weights_folder=None) -> None:
        """Upsample single/multi-band array using OpenCV.

        Args:
            weights_folder (_type_, optional): Path to models weights.
            Defaults to None.

        """
        self.model_name = model_name
        self._models = {k: {s: None for s in _MODELS_SCALES[k]}
                        for k in _MODELS
                        }

        if weights_folder is None:
            if _DZ_WEIGHTS.is_dir():
                weights_folder = _DZ_WEIGHTS
            else:
                weights_folder = os.envron.get(
                    'XDG_CACHE_HOME',
                    Path.home / '.cache/opencv_superres_models')
                if not weights_folder.is_dir():
                    weights_folder.mkdir(parents=True)

        self._weights_folder = Path(weights_folder)

    def _model(self, model_name, scale):
        cv_model = self._models[model_name][scale]
        if cv_model is None:
            cv_model = self._init_model(model_name, scale)
            self._models[model_name][scale] = cv_model
        return cv_model

    def _init_model(self, model_name, scale):
        import cv2

        if model_name not in _MODELS:
            raise ValueError(f"{model_name} not supported. Should be "
                             f"one of {_MODELS}")

        if scale not in (2, 3, 4, 8):
            raise ValueError("scale must be either 2, 4 or 8")

        if scale == 8 and model_name != 'lapsrn':
            raise ValueError("if scale is 8, model_name must be 'lapsrn'")
        if scale == 3 and model_name == 'lapsrn':
            raise ValueError("if scale is 3, model_name can't be 'lapsrn'")

        model_basepath = f'{MODELS_NAMES_CV[model_name]}_x{scale}.pb'

        weights_path = (self._weights_folder /
                        model_basepath)

        if not weights_path.is_file():
            self._download_weights(weights_path)

        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(str(weights_path))
        sr.setModel(model_name.replace('-small', ''), scale)
        return sr

    def _download_weights(self, weights_path):
        raise NotImplementedError("Please provide a valid weights_folder")

    def _upscale(self, cv_model, im, channel_dim=0):

        ndim = im.ndim

        if ndim > 2:
            im = np.moveaxis(im, channel_dim, 2)

        im = cv_model.upsample(im)

        if ndim > 2:
            im = np.moveaxis(im, 2, channel_dim)

        return im

    @staticmethod
    def _to_uint8(arr, channel_dim=0):

        chs = tuple([c for c in range(arr.ndim) if c != channel_dim])

        vmin = np.expand_dims(arr.min(axis=chs), chs)
        vmax = np.expand_dims(arr.max(axis=chs), chs)

        amin = np.broadcast_to(
            vmin,
            arr.shape)
        amax = np.broadcast_to(
            vmax,
            arr.shape)
        eps = 1e-6
        arr = (arr - amin) / (amax - amin + eps)

        arr *= 255
        arr = arr.astype(np.uint8)

        return arr, vmin, vmax

    def upscale(self, im, scale=4, progress_bar=True, channel_dim=0):
        """Upsample

        Args:
            im (ndarray): array to upscale
            model_name (str, optional): Can be one of lapsrn, edsr, espcn,
            fsrcnn-small, fsrcnn. Defaults to 'lapsrn'.
            scale (int, optional): Scaling factor. Can be 2, 3, 4, 8. 3 is not
            available for 'lapsrn' and 8 is only available for 'lapsrn'.
            Defaults to 4.
            channel_dim (int, optional): Bands channel if any. Defaults to 0.
            Ignored if array is 2D.

        Returns:
            ndarray: upscaled array
        """
        from tqdm import tqdm
        model_name = self.model_name
        cv_model = self._model(model_name, scale)

        if im.ndim == 2:
            im = np.expand_dims(im, channel_dim)

        im, vmin, vmax = self._to_uint8(im, channel_dim)

        if (im.ndim == 2) or (im.shape[channel_dim] == 3):
            new = self._upscale(cv_model, im, channel_dim)

        else:
            im = np.moveaxis(im, channel_dim, 0)
            c, h, w = im.shape

            new = np.zeros((c, h * scale, w * scale), dtype=np.uint8)

            ids = tqdm(range(c)) if progress_bar else range(c)

            for i in ids:
                new[i] = self._upscale(cv_model, im[i])

            new = np.moveaxis(new, 0, channel_dim)

        # restore original scaling
        new = new.astype(np.float32)
        new = new / 255
        vmin = np.broadcast_to(vmin, new.shape)
        vmax = np.broadcast_to(vmax, new.shape)
        new = new * (vmax - vmin) + vmin

        return new


def _contrast_stretch_percentile(img, pmin=2, pmax=98):
    from skimage import exposure
    p2, p98 = np.percentile(img, (pmin, pmax))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    return img_rescale


def _reduce_img(reducer, img, rescale=True):
    height4, width4, ch = img.shape
    vec = img.reshape((height4 * width4, ch))
    embedding = reducer.fit_transform(vec)

    labels = np.reshape(embedding, (height4, width4, embedding.shape[1]))

    if rescale:
        for i in range(labels.shape[2]):
            labels[:, :, i] = labels[:, :, i] - labels[:, :, i].min()
            labels[:, :, i] = labels[:, :, i] / labels[:, :, i].max()

    return labels
