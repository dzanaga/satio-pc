from setuptools import setup, find_packages
from satio_pc import __version__

tests_require = [
    'pytest>=4.5.0',
    'mock'
]

version = f"{__version__}"
setup(
    name="satio_pc",
    version=version,
    author="Daniele Zanaga",
    author_email="daniele.zanaga@vito.be",
    description=("Utilities for loading and processing of satellite data"),
    url='',
    license="Property of VITO NV",

    # adding packages
    packages=find_packages(include=['satio_pc*']),
    setup_requires=['pytest-runner'],
    tests_require=tests_require,
    package_data={
        '': ['layers/*'],
    },
    zip_safe=True,
    install_requires=[
        'boto3>=1.11.9',
        'geopandas>=0.7.0',
        'h5netcdf',
        'importlib-resources',
        'loguru>=0.5.3',
        'matplotlib>=3.5.3',
        'numba>=0.48.0',
        'joblib>=0.17.0',
        'openeo>=0.4.8',
        'psutil>=5.7.3',
        'rasterio>=1.1.1',
        'rio-cogeo>=3.4.1',
        'richdem',
        'scikit-image',
        'scikit-learn>=0.22.1',
        'shapely',
        'tqdm',
        'xarray>=0.12.3',
    ]
)
