import numpy as np

from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm, trange


def make_sample_n(sample_fn, parallel=True, pool=None):
    """Takes in a sample function and allows for parallelized sampling

    :param sample_fn: A function to compute samples
    :type sample_fn: function
    :param parallel: True if parallelization is to be used to compute samples, defaults to True
    :type parallel: bool, optional
    :param pool: Pool to use for parallelization if specified, defaults to None
    :type pool: multiprocessing.pool.Pool, optional
    """

    def sample_n(n, pool=pool):
        """Inner function to draw n samples using a specified sample function

        :param n: The number of samples to compute
        :type n: int
        :param pool: Pool to use for parallelization if specified, defaults to pool
        :type pool: multiprocessing.pool.Pool, optional
        :return: Array of n samples
        :rtype: numpy.ndarray
        """
        if parallel:
            if pool is None:
                print(f"Using {cpu_count()} CPUs")
                p = Pool(cpu_count())
            else:
                p = pool
            return np.array([s for s in tqdm(p.imap(sample_fn, np.arange(n)), total=n)])
        else:
            return np.array([sample_fn() for i in trange(n)])

    return sample_n
