"""
Adapted from https://github.com/sentinel-hub/eo-learn/blob/master/eolearn/coregistration/coregistration.py  # noqa
"""
import cv2
import numpy as np
from loguru import logger


def sobel(src: np.ndarray) -> np.ndarray:
    """Method which calculates and returns the gradients for the input image,
    which are better suited for co-registration
    """
    # Calculate the x and y gradients using Sobel operator
    src = src.astype(np.float32)

    grad_x = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)

    # Combine and return the two gradients
    return cv2.addWeighted(np.absolute(grad_x),
                           0.5,
                           np.absolute(grad_y),
                           0.5,
                           0)


def get_warp_matrix(im1,
                    im2,
                    number_of_iterations=5000,
                    termination_eps=1e-10,
                    max_translation=5):

    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                number_of_iterations,
                termination_eps)

    try:
        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv2.findTransformECC(im1, im2,
                                                 warp_matrix,
                                                 warp_mode, criteria)
    except Exception as e:
        logger.error(e)

    if is_translation_large(warp_matrix, max_translation):
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    return warp_matrix


def warp(im, warp_matrix):
    sz = im.shape
    im_warped = cv2.warpAffine(im,
                               warp_matrix,
                               (sz[1], sz[0]),
                               flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return im_warped


def is_translation_large(warp_matrix: np.ndarray,
                         max_translation: int = 5) -> bool:
    """Method that checks if estimated linear translation could be implausible.
    This function checks whether the norm of the estimated translation in
    pixels exceeds a predefined value.
    """
    d = np.linalg.norm(warp_matrix[:, 2]).astype(float)
    print(d)
    return d > max_translation


def warp_ts(ts, warp_matrices):
    t, b, y, x = ts.data.shape
    data = ts.data
    data_warped = np.zeros_like(data)
    for ti, warp_matrix in zip(range(t), warp_matrices):
        for bi in range(b):
            data_warped[ti][bi] = warp(data[ti][bi], warp_matrix)

    new_ts = ts.copy()
    new_ts.data = data_warped
    return new_ts


def coregister(ts,
               reference_band='B08',
               max_translation=3,
               mask_zeros=True):
    # Get reference band
    ref = ts.sel(band=[reference_band])

    # Compute median sobel gradient
    ref_med = ref.satio.percentile(q=[50])
    ref_sob = sobel(ref_med.isel(time=0).sel(band='s2-B08-p50').data)

    # compute gradients of time series
    sobs = np.array([sobel(ref.isel(time=i).data)
                     for i in range(ref.shape[0])])

    if mask_zeros:
        sobs_masks = ref.data > 0
        sobs = sobs * sobs_masks

    # Compute warp matrices
    warp_matrices = np.array([get_warp_matrix(ref_sob, sobs[i],
                                              max_translation=max_translation)
                              for i in range(sobs.shape[0])])

    # Warp time series
    ts_warped = warp_ts(ts, warp_matrices)

    return ts_warped
