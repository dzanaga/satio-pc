import os
from pathlib import Path

from loguru import logger
import torch
import xarray as xr
import numpy as np
from tqdm import tqdm
from super_image import (DrlnModel, MdsrModel, EdsrModel, MsrnModel,
                         A2nModel, PanModel)

MODELS_NAMES_SUPERIMAGE = {
    'drln': DrlnModel,
    'drln-bam': DrlnModel,
    'mdsr': MdsrModel,
    'edsr-base': EdsrModel,
    'edsr': EdsrModel,
    'msrn': MsrnModel,
    'a2n': A2nModel,
    'pan': PanModel,
}

MODELS_CACHE_DIR = Path(
    "/data/users/Public/dzanaga/models_weights/")


def download_model_data(model_name, scale, cache_dir, overwrite=False):
    model_dir = Path(cache_dir) / model_name
    if model_name not in MODELS_NAMES_SUPERIMAGE:
        raise ValueError(f"Invalid model name: {model_name}")

    base_url = f"https://huggingface.co/eugenesiow/{model_name}/resolve/main/"
    files = ["config.json", f"pytorch_model_{scale}x.pt"]

    files_to_download = [base_url + file for file in files]
    dst_filenames = [model_dir / file for file in files]
    # check if files exist otherwise download them

    for file_to_download, dst_filename in zip(files_to_download,
                                              dst_filenames):
        if dst_filename.exists() and not overwrite:
            logger.info(f"File {dst_filename} already exists. "
                        "Skipping download.")
        else:
            logger.info(f"Downloading {file_to_download} to {dst_filename}")
            download_file(file_to_download, dst_filename)


def download_file(url, dst_filename):
    import requests
    with requests.get(url) as r:
        r.raise_for_status()

        dst_filename.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_filename, 'wb') as f:
            f.write(r.content)


def get_cache_dir(cache_dir=None):
    """
    Returns the cache directory for the super_image module.
    If `cache_dir` is not provided, the default cache directory is used.
    """
    if cache_dir is None:
        cache_dir = (MODELS_CACHE_DIR if MODELS_CACHE_DIR.is_dir()
                     else os.environ.get("XDG_CACHE_DIR",
                                         Path.home() / ".cache"))
        cache_dir = Path(cache_dir) / "super_image"

    return cache_dir


class SuperImage:

    def __init__(self, model_name='edsr-base', cache_dir=None):
        """
        Initializes a SuperImage object.

        Args:
            model_name (str): The name of the super-resolution model to
            use. Available models: 'drln', 'drln-bam', 'mdsr', 'edsr-base',
            'edsr', 'msrn', 'a2n', 'pan'. Default is 'edsr-base'.
            cache_dir (str): The directory to use for caching model files.
            If None, the default cache directory will be used.
        """
        self.model_name = model_name
        self._cache_dir = get_cache_dir(cache_dir)
        self._models = {}

    def __repr__(self):
        return f"SuperImage(model_name={self.model_name}, " \
               f"cache_dir={self._cache_dir})"

    def __str__(self):
        return f"SuperImage(model_name={self.model_name}, " \
               f"cache_dir={self._cache_dir})"

    def _initialize_model(self, scale):
        if self.model_name not in MODELS_NAMES_SUPERIMAGE:
            raise ValueError(f"Invalid model name: {self.model_name}")

        # model_path = download_model_data(self.model_name, scale,
        #                                  self._cache_dir)

        model_path = f'eugenesiow/{self.model_name}'
        return MODELS_NAMES_SUPERIMAGE[self.model_name].from_pretrained(
            model_path,
            scale=scale,
            cache_dir=self._cache_dir
        )

    def model(self, scale=4):
        model = self._models.get(scale, None)
        if model is None:
            model = self._initialize_model(scale)
            self._models[scale] = model
        return model

    def _normalize_channelwise(self, arr):
        arr_min = arr.min(axis=(1, 2), keepdims=True)
        arr_max = arr.max(axis=(1, 2), keepdims=True)
        d = (arr_max - arr_min)
        arr = (arr - arr_min) / np.where(d == 0, 1, d)
        return arr, arr_min, arr_max

    def _normalize_global(self, arr):
        arr_min = arr.min()
        arr_max = arr.max()
        d = (arr_max - arr_min)
        arr = (arr - arr_min) / np.where(d == 0, 1, d)
        return arr, arr_min, arr_max

    def _preprocess(self, arr, normalize_method='channelwise'):

        if isinstance(arr, xr.DataArray):
            arr = arr.data

        if normalize_method == 'channelwise':
            arr, arr_min, arr_max = self._normalize_channelwise(arr)
        elif normalize_method == 'global':
            arr, arr_min, arr_max = self._normalize_global(arr)
        else:
            raise ValueError(f"Invalid normalize_method: {normalize_method}")

        return arr, arr_min, arr_max

    def upscale(self, arr, scale=4, normalize_method='channelwise',
                progress_bar=True):

        model = self.model(scale)
        arr, arr_min, arr_max = self._preprocess(arr, normalize_method)

        # Chunk the array in groups of 3
        n_bands = arr.shape[0]
        n_chunks = n_bands // 3
        last_chunk_size = n_bands % 3
        if last_chunk_size > 0:
            arr = np.pad(arr, ((0, 3 - last_chunk_size), (0, 0), (0, 0)),
                         mode='constant')
            n_chunks += 1
        arr_chunks = np.array_split(arr, n_chunks, axis=0)

        # Compute the chunked preds
        preds_chunks = []
        if progress_bar:
            arr_chunks = tqdm(arr_chunks)
        with torch.no_grad():
            for chunk in arr_chunks:
                chunk = torch.from_numpy(chunk).float()
                chunk = chunk.unsqueeze(0)
                preds_chunk = model(chunk)
                preds_chunks.append(np.squeeze(preds_chunk.numpy()))

        # Stack the chunked preds back together
        preds = np.concatenate(preds_chunks, axis=0)
        if last_chunk_size > 0:
            preds = preds[:n_bands]

        # Restore the original values by doing the inverse of the normalization
        if normalize_method == 'channelwise':
            preds = preds * (arr_max - arr_min) + arr_min
        elif normalize_method == 'global':
            preds = preds * (arr_max - arr_min) + arr_min * np.ones_like(preds)

        return preds
