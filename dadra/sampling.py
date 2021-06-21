import math
import numpy as np

from multiprocessing import Pool, cpu_count
from numpy.random import default_rng
from scipy.integrate import odeint


def num_samples(epsilon, delta, n_x, const=None):
    """
    Compute the number of samples needed to satisfy the specified probabilistic guarantees.

    :param epsilon: The accuracy parameter.
    :type epsilon: float

    :param delta: The confidence parameter.
    :type delta: float

    :param n_x: The state dimension.
    :type n_x: int

    :param const: The constraints placed on the parameters A and b
    :type const: string
    """
    if const is None:
        n_theta = 0.5 * (n_x ** 2 + 3 * n_x)
    elif const == "diagonal":
        n_theta = 2 * n_x
    elif const == "scalar":
        n_theta = 1
    N = math.ceil(math.e * (math.log(1 / delta) + n_theta) / (epsilon * (math.e - 1)))
    return N


def lorenz_system(z, t, sigma=10, rho=28, beta=8 / 3):
    """
    Define the dynamics of the lorenz system based on initial states.

    :param z: The initial states of the lorenz system (3 dimensions).
    :type z: numpy.ndarray

    :param t: The time at which the derivate of the system is computed.
    :type t: int

    :param sigma: Lorenz system constant sigma
    :type sigma: float

    :param rho: Lorenz system constant rho
    :type rho: float

    :param beta: Lorenz system constant beta
    :type beta: float
    """
    dzdt = [
        sigma * (z[1] - z[0]),
        z[0] * (rho - z[2]) - z[1],
        z[0] * z[1] - beta * z[2],
    ]
    return dzdt


def sample_lorenz(timesteps=100):
    """
    Obtain sample from Lorenz system over specified number of timesteps.

    :param timesteps: The number of timesteps over which to compute the sample.
    :type timesteps: int
    """
    t = np.linspace(0, timesteps, 1001)
    ru = default_rng().uniform
    z0 = np.array([ru(0, 1), ru(0, 1), ru(0, 1)])
    sol = odeint(lorenz_system, z0, t)
    return sol[-1]


def make_sample_n(sample_fn, parallel=True, pool=None):
    """
    Takes in a sample function and allows for parallelized sampling.

    :param sample_fn: A function to compute samples.
    :type sample_fn: function

    :param parallel: True if parallelization is to be used to compute samples.
    :type parallel: bool

    :param pool: Pool to use for parallelization if specified.
    :type pool: multiprocessing.pool.Pool
    """

    def sample_n(n, pool=pool):
        """
        Inner function to draw n samples using a specified sample function.

        :param n: The number of samples to compute.
        :type n: int

        :param pool: Pool to use for parallelization if specified.
        :type pool: multiprocessing.pool.Pool
        """
        if parallel:
            if pool is None:
                print(f"Using {cpu_count()} CPUs")
                p = Pool(cpu_count())
            else:
                p = pool
            return np.array(list(p.map(sample_fn, np.arange(n))))
        else:
            return np.array([sample_fn() for i in range(n)])

    return sample_n


sample_lorenz_n = make_sample_n(sample_lorenz)
"""
Function to sample from Lorenz system.

:param n: The number of samples to compute.
:type n: int
"""
