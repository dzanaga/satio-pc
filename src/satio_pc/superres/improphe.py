from numba import jit, prange
import numpy as np

@jit(nopython=True)
def rescale_weight(weight, minWeight, maxWeight):
    if weight == 0:
        normweight = 1.0
    elif minWeight == maxWeight:
        normweight = 1.0
    else:
        normweight = 1.0 + np.exp(25 * ((weight - minWeight) / (maxWeight - minWeight)) - 7.5)
    return normweight

@jit(nopython=True)
def distance_kernel(nk):
    kernel = np.empty((nk,nk), dtype=np.float32)
    h = (nk-1)//2
    
    ki = -1
    for ii in range(-h, h+1):
        ki += 1
        kj = -1
        for jj in range(-h, h+1):
            kj += 1
            kernel[ki,kj] = np.sqrt(ii**2 + jj**2)
            
    return kernel

@jit(nopython=True)
def improphe(toa, bands_m, bands_c, mink, kSize=5):
    h = (kSize - 1) // 2
    KDIST = distance_kernel(kSize)
    ny, nx = toa.shape[1:]
    ns = 5
    Sthr = np.array([0.0025, 0.005, 0.01, 0.02, 0.025])
    predict = np.copy(toa)

    for i in prange(ny):
        for j in prange(nx):
            if toa[bands_m[0], i, j] != np.nan:
                wn = 0
                Sclass = np.zeros(KDIST.size, dtype=np.int32)
                Srecord = np.zeros(KDIST.size, dtype=np.float32)
                # Srange = np.array([np.full(ns, np.inf, dtype=np.float32), np.full(ns, -np.inf, dtype=np.float32)])
                Srange = np.empty((2, ns), dtype=np.float32)
                Srange[0, :] = np.inf
                Srange[1, :] = -np.inf
                coarsePixels = np.empty((KDIST.size, len(bands_c)), dtype=np.float32)
                Sn = np.zeros(ns, dtype=np.int32)
                ki = -1
                for ii in range(-h, h+1):
                    ki += 1
                    kj = -1
                    for jj in range(-h, h+1):
                        kj += 1
                        ni, nj = i + ii, j + jj
                        if 0 <= ni < ny and 0 <= nj < nx and KDIST[ki, kj] <= h and toa[bands_m[0], ni, nj] != np.nan:
                            S = np.sum(np.abs((toa[bands_m, i, j] - toa[bands_m, ni, nj]))) / len(bands_m)
                            Sc = np.where(Sthr > S)[0][0] if np.any(Sthr > S) else -1
                            if Sc >= 0:
                                Sclass[wn] = Sc
                                Srecord[wn] = S
                                if S > 0:
                                    for s in range(Sc, ns):
                                        if S > Srange[1, s]: Srange[1, s] = S
                                        if S < Srange[0, s]: Srange[0, s] = S
                                coarsePixels[wn] = toa[bands_c, ni, nj]
                                wn += 1
                                for s in range(Sc, ns):
                                    Sn[s] += 1
                if wn != 0:
                    Sc = np.where(Sn >= mink)[0][0] if np.any(Sn >= mink) else ns - 1
                    weight = 0.0
                    weightxdata = np.zeros(len(bands_c), dtype=np.float32)
                    for k in range(wn):
                        if Sclass[k] <= Sc:
                            SS = rescale_weight(Srecord[k], Srange[0, Sc], Srange[1, Sc])
                            W = 1.0 / SS
                            weightxdata += W * coarsePixels[k]
                            weight += W
                    for f, index_c in enumerate(bands_c):
                        predict[index_c, i, j] = (weightxdata[f] / weight)
    return predict
