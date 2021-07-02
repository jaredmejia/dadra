import numpy as np

from dadra.disturbance import Disturbance


def duffing_oscillator(
    y, t, disturbance: Disturbance = None, epsilon=1.0, alpha=0.05, omega=1.3, gamma=0.4
):
    """Defines the dynamics of a Duffing oscillator based on initial states

    :param y: The initial states of the Duffing oscillator (2 dimensions)
    :type y: numpy.ndarray
    :param t: The time at which the derivatives of the system are computed
    :type t: int
    :param disturbance: The disturbance to be added to each dimension, defaults to None
    :type disturbance: :class:`dadra.Disturbance`, optional
    :param epsilon: The weight of the disturbance to be added to each dimension, defaults to 1.0
    :type epsilon: float, optional
    :param alpha: Duffing oscillator constant alpha, defaults to 0.05
    :type alpha: float, optional
    :param omega: Duffing oscillator constant omega, defaults to 1.3
    :type omega: float, optional
    :param gamma: Duffing oscillator constant gamma, defaults to 0.4
    :type gamma: float, optional
    :return: The partial derivatives of the Duffing oscillator along x and y
    :rtype: list
    """
    if disturbance is None:
        dydt = [y[1], -alpha * y[1] + y[0] - y[0] ** 3 + gamma * np.cos(omega * t)]
    else:
        dydt = [
            y[1] + disturbance.get_dist(0, t) * epsilon,
            -alpha * y[1]
            + y[0]
            - y[0] ** 3
            + gamma * np.cos(omega * t)
            + disturbance.get_dist(1, t) * epsilon,
        ]

    return dydt


def lorenz_system(
    z, t, disturbance: Disturbance = None, epsilon=1.0, sigma=10.0, rho=28.0, beta=8 / 3
):
    """Defines the dynamics of a Lorenz system based on initial states

    :param z: The initial states of the Lorenz system (3 dimensions)
    :type z: numpy.ndarray
    :param t: The time at which the derivatives of the system are computed
    :type t: int
    :param disturbance: The disturbance to be added to each dimension, defaults to None
    :type disturbance: :class:`dadra.Disturbance`, optional
    :param epsilon: The weight of the disturbance to be added to each dimension, defaults to 1.0
    :param sigma: Lorenz system constant sigma, defaults to 10.
    :type sigma: float, optional
    :param rho: Lorenz system constant rho, defaults to 28.
    :type rho: float, optional
    :param beta: Lorenz system constant beta, defaults to 8/3
    :type beta: float, optional
    :return: The partial derivatives of the Lorenz system along x, y, and z
    :rtype: list
    """
    if disturbance is None:
        dzdt = [
            sigma * (z[1] - z[0]),
            z[0] * (rho - z[2]) - z[1],
            z[0] * z[1] - beta * z[2],
        ]
    else:
        dzdt = [
            sigma * (z[1] - z[0]) + disturbance.get_dist(0, t) * epsilon,
            z[0] * (rho - z[2]) - z[1] + disturbance.get_dist(1, t) * epsilon,
            z[0] * z[1] - beta * z[2] + disturbance.get_dist(2, t) * epsilon,
        ]
    return dzdt


def planar_quadrotor(y, t, u1, u2, g=9.81, K=0.89 / 1.4, d0=70, d1=17, n0=55):
    """Defines the dynamics of a Planar Quadrotor Model based on initial states

    :param y: The initial states of the model (6 states: x, h, theta, and their first derivatives)
    :type y: numpy.ndarray
    :param t: The time at which the derivatives of the system are computed
    :type t: int
    :param u1: System input 1, treated as disturbance, the motor thrust
    :type u1: float
    :param u2: System input 2, treated as disturbance, the desired angle
    :type u2: float
    :param g: System constant g, defaults to 9.81
    :type g: float, optional
    :param K: System constant K, defaults to 0.89/1.4
    :type K: float, optional
    :param d0: System constant d0, defaults to 70
    :type d0: int, optional
    :param d1: System constant d1, defaults to 17
    :type d1: int, optional
    :param n0: System constant n0, defaults to 55
    :type n0: int, optional
    :return: The partial derivatives of the Planar Quadrotor Model system along each of the 6 states.
    :rtype: list
    """
    x, xdot, h, hdot, theta, thetadot = y
    dydt = [
        xdot,
        u1 * K * np.sin(theta),
        hdot,
        -g + u1 * K * np.cos(theta),
        thetadot,
        -d0 * theta - d1 * thetadot + n0 * u2,
    ]
    return dydt


def traffic(y, t, d=0):
    """Defines the dynamics of a Monotone Traffic Model based on initial states

    :param y: The initial states of the model (n-dimensional)
    :type y: numpy.ndarray
    :param t: The time at which the derivatives of the system are computed
    :type t: int
    :param d: disturbances, defaults to 0
    :type d: int, optional
    :return: The partial derivatives of the Monotone Traffic Model along each of the n dimensions
    :rtype: numpy.ndarray
    """
    v = 0.5  # free-flow speed, in links/period
    w = 1 / 6  # congestion-wave speed, in links/period
    c = 40  # capacity (max downstream flow), in vehicles/period
    xbar = 320  # max occupancy when jammed, in vehicles
    b = 1  # fraction of vehicle staying on the network after each link
    T = 30  # time step for the continuous-time model
    nx = np.size(y)
    dy = np.zeros(np.shape(y))
    dy[0] = 1 / T * (d - min(c, v * y[0], 2 * w * (xbar - y[1])))
    for i in range(1, nx - 1):
        dy[i] = (
            1
            / T
            * (
                b * min(c, v * y[i - 1], w * (xbar - y[i]))
                - min(c, v * y[i], w / b * (xbar - y[i + 1]))
            )
        )
    dy[-1] = 1 / T * (b * min(c, v * y[-2], w * (xbar - y[-1])) - min(c, v * y[-1]))
    return dy
