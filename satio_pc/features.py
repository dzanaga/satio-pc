import dask.array as da
import xarray as xr
import numpy as np


def percentile(ts, q=[10, 25, 50, 75, 90], name_prefix='s2'):

    chunks = list(ts.chunks)

    chunks[0] = (len(q),) * len(chunks[1])

    p = da.map_blocks(
        np.percentile,
        ts.data,
        q,
        chunks=chunks,
        axis=0).astype(np.float32)

    p = da.concatenate([p[:, b, :, :] for b in range(ts.band.size)],
                       axis=0)
    p_names = [f'{name_prefix}-{b.item()}-p{qi}' for b in ts.band for qi in q]

    p = da.expand_dims(p, axis=0)
    p = xr.DataArray(p,
                     dims=('time', 'band', 'y', 'x'),
                     coords={'time': ts.time[[0]],
                             'band': p_names,
                             'y': ts.y,
                             'x': ts.x},
                     attrs=ts.attrs)

    return p
