
import tempfile
import numpy as np
from loguru import logger

from satio_pc.preprocessing.timer import FeaturesTimer


def force_unique_time(darr):
    """Add microseconds to time vars which repeats in order to make the
    time index of the DataArray unique, as sometimes observations from the same
    day can be split in multiple obs"""
    unique_ts, counts_ts = np.unique(darr.time, return_counts=True)
    double_ts = unique_ts[np.where(counts_ts > 1)]

    new_time = []
    c = 0
    for i in range(darr.time.size):
        v = darr.time[i].values
        if v in double_ts:
            v = v + c
            c += 1
        new_time.append(v)
    new_time = np.array(new_time)
    darr['time'] = new_time
    return darr


def load_gamma0(bounds,
                epsg,
                time_range):
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

    search = catalog.search(
        collections=[collection],
        bbox=ll_bounds,
        datetime=time_range
    )

    items = search.item_collection()
    if len(items) == 0:
        print("no overlaps found")

    stack = stackstac.stack(items,
                            assets=['vv', 'vh'],
                            epsg=f'EPSG:{epsg}',
                            bounds=bounds,
                            xy_coords='center',
                            resolution=10,
                            dtype=np.float32)

    del stack.attrs['spec']

    ds_vars = list(stack.coords.keys())

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
    stack = stack.drop_vars(drop_vars)
    stack = force_unique_time(stack)

    str_vars = ['id',
                'end_datetime',
                'start_datetime',
                'sat:absolute_orbit',
                'sat:orbit_state',
                'sat:relative_orbit',
                's1:datatake_id',]
    for v in str_vars:
        stack[v] = stack[v].astype(str)

    return stack


def preprocess_gamma0(stack,
                      start_date,
                      end_date,
                      composite_freq=10,
                      composite_window=20,
                      speckle_kwargs=None,
                      multitemp_speckle=True,
                      tmpdir='.'):

    if speckle_kwargs is None:
        speckle_kwargs = dict(kernel='gamma',
                              mtwin=15,
                              enl=7)

    timer10 = FeaturesTimer(10)

    with tempfile.TemporaryDirectory(prefix='satio_tmp-', dir=tmpdir) as tmpdirname:

        # download
        logger.info("Loading block data")
        timer10.load.start()
        stack = stack.satio.cache(tmpdirname.name)
        timer10.load.stop()

        # 10m
        # speckle
        logger.info("Applying multi-temporal speckle filter")
        if multitemp_speckle:
            timer10.speckle.start()
            stack_fil = (stack.satio.multitemporal_speckle(**speckle_kwargs)
                         .satio.cache(tmpdirname.name))
            timer10.speckle.stop()
        else:
            stack_fil = stack

        logger.info("Compositing 10m block data")
        # composite
        stack_comp = stack_fil.satio.composite(
            freq=composite_freq,
            window=composite_window,
            start=start_date,
            end=end_date).satio.cache(tmpdirname.name)
        timer10.composite.stop()

        logger.info("Interpolating 10m block data")
        # interpolation
        timer10.interpolate.start()
        stack_interp = stack_comp.satio.interpolate(
        ).satio.cache()
        timer10.interpolate.stop()

        timer10.load.log()
        timer10.speckle.log()
        timer10.composite.log()
        timer10.interpolate.log()

    return stack_interp
