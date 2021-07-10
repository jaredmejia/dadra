import numpy as np

from dadra.disturbance import Disturbance
from dadra.sampling import make_sample_n
from dadra.systems.dynamics import *
from numpy.random import default_rng
from scipy.integrate import odeint


def sample_oscillator(
    x=None, disturbance: Disturbance = None, timesteps=100, parts=1001
):
    """Obtains a sample from a Duffing Oscillator over the specified number of timesteps

    :param x: Placeholder variable over which to evaluate, defaults to None
    :type x: NoneType, optional
    :param disturbance: The disturbance to be added to each dimension, defaults to None
    :type disturbance: :class:`dadra.Disturbance`, optional
    :param timesteps: The number of timesteps over which to compute the sample, defaults to 100
    :type timesteps: int, optional
    :param parts: The number of parts to partition the time interval into for computing the sample, defaults to 1001
    :type parts: int, optional
    :return: The sample from the Duffing Oscillator at the last timestep
    :rtype: numpy.ndarray
    """
    t = np.linspace(0, timesteps, parts)
    y0 = np.array([np.random.uniform(0.95, 1.05), np.random.uniform(-0.05, 0.05)])
    if disturbance is None:
        sol = odeint(duffing_oscillator, y0, t)
    else:
        disturbance.draw_alphas()
        sol = odeint(duffing_oscillator, y0, t, args=tuple([disturbance]))
    return sol[-1]


def sample_lorenz(x=None, disturbance: Disturbance = None, timesteps=100, parts=1001):
    """Obtains a sample from a Lorenz system over the specified number of timesteps

    :param x: Placeholder variable over which to evaluate, defaults to None
    :type x: NoneType, optional
    :param disturbance: The disturbance to be added to each dimension, defaults to None
    :type disturbance: :class:`dadra.Disturbance`, optional
    :param timesteps: The number of timesteps over which to compute the sample, defaults to 100
    :type timesteps: int, optional
    :param parts: The number of parts to partition the time interval into for computing the sample, defaults to 1001
    :type parts: int, optional
    :return: The sample from the Lorenz system at the last timestep
    :rtype: numpy.ndarray
    """
    t = np.linspace(0, timesteps, parts)
    ru = default_rng().uniform
    z0 = np.array([ru(0, 1), ru(0, 1), ru(0, 1)])
    if disturbance is None:
        sol = odeint(lorenz_system, z0, t)
    else:
        disturbance.draw_alphas()
        sol = odeint(lorenz_system, z0, t, args=tuple([disturbance]))
    return sol[-1]


def sample_planar_quadrotor(x=None, timesteps=5, parts=5001):
    """Obtains a sample from a Planar Quadrotor Model over the specified number of timesteps

    :param x: Placeholder variable over which to evaluate, defaults to None
    :type x: NoneType, optional
    :param timesteps: The number of timesteps over which to compute the sample, defaults to 5
    :type timesteps: int, optional
    :param parts: The number of parts to partition the time interval into for computing the sample, defaults to 5001
    :type parts: int, optional
    :return: The sample from the Planar Quadrotor Model at the last timestep
    :rtype: numpy.ndarray
    """
    g = 9.81
    K = 0.89 / 1.4
    ru = np.random.uniform
    t = np.linspace(0, timesteps, parts)

    # initial states
    y0 = np.array(
        [
            ru(-1.7, 1.7),
            ru(-0.8, 0.8),
            ru(0.3, 2.0),
            ru(-1.0, 1.0),
            ru(-np.pi / 12, np.pi / 12),
            ru(-np.pi / 2, np.pi / 2),
        ]
    )
    sol = odeint(
        planar_quadrotor,
        y0,
        t,
        args=(ru(-1.5 + g / K, 1.5 + g / K), ru(-np.pi / 4, np.pi / 4)),
    )
    return sol[-1]


def sample_traffic(x=None, nx=6, timesteps=100, parts=10001):
    """Obtains a sample from a Monotone Traffic Model over the specified number of timesteps

    :param x: Placeholder variable over which to evaluate, defaults to None
    :type x: NoneType, optional
    :param nx: The state dimension of the model, defaults to 6
    :type nx: int, optional
    :param timesteps: The number of timesteps over which to compute the sample, defaults to 100
    :type timesteps: int, optional
    :param parts: The number of parts to partition the time interval into for computing the sample, defaults to 10001
    :type parts: int, optional
    :return: The sample from the Monotone Traffic Model at the last timestep
    :rtype: numpy.ndarray
    """
    t = np.linspace(0, timesteps, parts)
    y0 = 100 * np.ones(nx) + 100 * np.random.rand(nx)
    sol = odeint(traffic, y0, t)
    return sol[-1]


def sample_quadrotor(x=None, timesteps=5, parts=5001):
    """Obtains sample from the 12-state Quadrotor Model across all timesteps

    :param x: Placeholder variable over which to evaluate, defaults to None
    :type x: NoneType, optional
    :param timesteps: The number of timesteps over which to compute the sample, defaults to 5
    :type timesteps: int, optional
    :param parts: The number of parts to partition the time interval into for computing the sample, defaults to 5001
    :return: The sample from the 12-state Quadrotor Model accross all timesteps
    :rtype: numpy.ndarray
    """
    t = np.linspace(0, timesteps, parts)
    ru = default_rng().uniform
    x1_0, x2_0, x3_0, x4_0, x5_0, x6_0 = [ru(-0.4, 0.4) for _ in range(6)]
    rand_states = np.array([x1_0, x2_0, x3_0, x4_0, x5_0, x6_0])
    x_0 = np.concatenate((rand_states, np.zeros(6)))
    sol = odeint(quadrotor, x_0, t)
    return sol


sample_oscillator_n = make_sample_n(sample_oscillator)
"""Function to sample from a Duffing Oscillator

:param n: The number of samples to compute.
:type n: int
"""


sample_lorenz_n = make_sample_n(sample_lorenz)
"""Function to sample from a Lorenz system

:param n: The number of samples to compute.
:type n: int
"""

sample_planar_quadrotor_n = make_sample_n(sample_planar_quadrotor)
"""Function to sample from a Planar Quadrotor Model

:param n: The number of samples to compute.
:type n: int
"""


sample_traffic_n = make_sample_n(sample_traffic)
"""Function to sample from a Monotone Traffic Model

:param n: The number of samples to compute.
:type n: int
"""

sample_quadrotor_n = make_sample_n(sample_quadrotor)
"""Function to sample from a 12-state Quadrotor Model

:param n: The number of samples to compute.
:type n: int
"""
