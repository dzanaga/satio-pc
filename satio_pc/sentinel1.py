
import tempfile
import numpy as np
import dask.array as da
import xarray as xr
from loguru import logger

from satio_pc.preprocessing.timer import FeaturesTimer
from satio_pc.sentinel2 import force_unique_time
from satio_pc.errors import NoGamma0Products


def lin_to_db(lin):
    '''
    Helper function to transform dB to power units
    '''
    return 10 * np.log10(lin)


def db_to_lin(db):
    '''
    Helper function to transform power to dB units
    '''
    return np.power(10, db / 10)


def load_gamma0(bounds,
                epsg,
                start_date,
                end_date):
    import stackstac
    import pystac_client
    import planetary_computer as pc
    from pyproj import Transformer

    transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
    ll_bounds = transformer.transform_bounds(*bounds)

    collection = 'sentinel-1-rtc'

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )

    time_range = f"{start_date}/{end_date}"

    search = catalog.search(
        collections=[collection],
        bbox=ll_bounds,
        datetime=time_range
    )

    items = search.item_collection()
    if len(items) == 0:
        raise NoGamma0Products("No data for given bounds")

    s1 = stackstac.stack(items,
                         assets=['vv', 'vh'],
                         epsg=f'EPSG:{epsg}',
                         bounds=bounds,
                         xy_coords='center',
                         resolution=10,
                         dtype=np.float32)

    del s1.attrs['spec']

    ds_vars = list(s1.coords.keys())

    keep_vars = ['time',
                 'id',
                 'band',
                 'x',
                 'y',
                 'end_datetime',
                 'start_datetime',
                 'sat:absolute_orbit',
                 'sat:orbit_state',
                 'sat:relative_orbit',
                 's1:datatake_id',
                 'epsg']

    drop_vars = [v for v in ds_vars if v not in keep_vars]
    s1 = s1.drop_vars(drop_vars)
    s1 = force_unique_time(s1)

    str_vars = ['id',
                'end_datetime',
                'start_datetime',
                'sat:absolute_orbit',
                'sat:orbit_state',
                'sat:relative_orbit',
                's1:datatake_id']
    for v in str_vars:
        s1[v] = s1[v].astype(np.dtype('U25'))

    return s1


def count_gamma0_obs(s1):
    obs_gamma0 = (~da.isnan(s1.isel(band=0))).sum(axis=0).compute()
    return obs_gamma0


def preprocess_gamma0(s1,
                      start_date,
                      end_date,
                      composite_freq=10,
                      composite_window=20,
                      composite_mode='mean',
                      multitemp_speckle=True,
                      speckle_kernel='gamma',
                      speckle_mtwin=15,
                      speckle_enl=7,
                      tmpdir='.'):

    speckle_kwargs = dict(kernel=speckle_kernel,
                          mtwin=speckle_mtwin,
                          enl=speckle_enl)

    timer10 = FeaturesTimer(10, 'gamma0')

    with tempfile.TemporaryDirectory(prefix='ewc_tmp-', dir=tmpdir) as \
            tmpdirname:

        # download
        logger.info("Loading block data")
        timer10.load.start()
        s1 = s1.ewc.cache(tmpdirname)
        # count obs
        obs_gamma0 = count_gamma0_obs(s1)
        timer10.load.stop()

        # 10m
        # speckle
        logger.info("Applying multi-temporal speckle filter")
        if multitemp_speckle:
            timer10.speckle.start()
            s1_fil = (s1.ewc.multitemporal_speckle(**speckle_kwargs)
                      .ewc.cache(tmpdirname))
            timer10.speckle.stop()
        else:
            s1_fil = s1

        logger.info("Compositing 10m block data")
        timer10.composite.start()
        s1_comp = s1_fil.ewc.composite(
            freq=composite_freq,
            window=composite_window,
            start=start_date,
            end=end_date).ewc.cache(tmpdirname)
        timer10.composite.stop()

        logger.info("Interpolating 10m block data")
        # interpolation
        timer10.interpolate.start()
        s1_interp = s1_comp.ewc.interpolate(
        ).ewc.cache(tmpdir)
        timer10.interpolate.stop()

        timer10.load.log()
        timer10.speckle.log()
        timer10.composite.log()
        timer10.interpolate.log()

        timer10.log()

    return s1_interp, obs_gamma0
