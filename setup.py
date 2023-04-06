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
        'boto3',
        'dask',
        'dask_image',
        'geopandas',
        'h5netcdf',
        'importlib-resources',
        'loguru',
        'rasterio',
        'rio-cogeo',
        'shapely',
        'tqdm',
    ],
    entry_points={
        "console_scripts": [
            "ewc = satio_pc.cli:ewc_cli",
        ]}
)
