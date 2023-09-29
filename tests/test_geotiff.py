import pytest
import numpy as np

from satio_pc.geotiff import compute_pixel_coordinates


def test_compute_pixel_coordinates_consistency():
    # Test with consistent bounds and shape
    bounds = (0, 0, 10, 10)
    shape = (10, 10)
    y_coords, x_coords = compute_pixel_coordinates(bounds, shape)
    assert len(y_coords) == shape[0]
    assert len(x_coords) == shape[1]


def test_compute_pixel_coordinates_resolution_error():
    # Test with inconsistent bounds and shape
    bounds = (0, 0, 10, 10)
    shape = (5, 10)  # Different number of rows
    with pytest.raises(ValueError):
        compute_pixel_coordinates(bounds, shape)


def test_compute_pixel_coordinates_coordinates():
    # Test with specific bounds and shape
    bounds = (0, 0, 10, 10)
    shape = (10, 10)
    y_coords, x_coords = compute_pixel_coordinates(bounds, shape)
    assert np.allclose(y_coords, np.linspace(9.5, 0.5, 10))
    assert np.allclose(x_coords, np.linspace(0.5, 9.5, 10))
