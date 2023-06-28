import concurrent.futures
import tqdm


def parallelize(f,
                my_iter,
                max_workers=4,
                progressbar=True,
                use_process_pool=False):

    if use_process_pool:
        Pool = concurrent.futures.ProcessPoolExecutor
    else:
        Pool = concurrent.futures.ThreadPoolExecutor

    with Pool(max_workers=max_workers) as ex:
        if progressbar:
            results = list(tqdm(ex.map(f, my_iter), total=len(my_iter)))
        else:
            results = list(ex.map(f, my_iter))
    return results
