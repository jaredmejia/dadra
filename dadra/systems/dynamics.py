import numpy as np

from dadra.disturbance import Disturbance


def duffing_oscillator(
    y, t, disturbance: Disturbance = None, eta=1.0, alpha=0.05, omega=1.3, gamma=0.4
):
    """Defines the dynamics of a Duffing oscillator based on initial states

    :param y: The initial states of the Duffing oscillator (2 dimensions)
    :type y: numpy.ndarray
    :param t: The time at which the derivatives of the system are computed
    :type t: int
    :param disturbance: The disturbance to be added to each dimension, defaults to None
    :type disturbance: :class:`dadra.Disturbance`, optional
    :param eta: The weight of the disturbance to be added to each dimension, defaults to 1.0
    :type eta: float, optional
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
            y[1] + disturbance.get_dist(0, t) * eta,
            -alpha * y[1]
            + y[0]
            - y[0] ** 3
            + gamma * np.cos(omega * t)
            + disturbance.get_dist(1, t) * eta,
        ]

    return dydt


def lorenz_system(
    z, t, disturbance: Disturbance = None, eta=1.0, sigma=10.0, rho=28.0, beta=8 / 3
):
    """Defines the dynamics of a Lorenz system based on initial states

    :param z: The initial states of the Lorenz system (3 dimensions)
    :type z: numpy.ndarray
    :param t: The time at which the derivatives of the system are computed
    :type t: int
    :param disturbance: The disturbance to be added to each dimension, defaults to None
    :type disturbance: :class:`dadra.Disturbance`, optional
    :param eta: The weight of the disturbance to be added to each dimension, defaults to 1.0
    :type eta: float, optional
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
            sigma * (z[1] - z[0]) + disturbance.get_dist(0, t) * eta,
            z[0] * (rho - z[2]) - z[1] + disturbance.get_dist(1, t) * eta,
            z[0] * z[1] - beta * z[2] + disturbance.get_dist(2, t) * eta,
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


def quadrotor(x, t, g=9.81, R=0.1, l=0.5, M_rotor=0.1, M=1.0, u1=1.0, u2=0.0, u3=0.0):
    """Defines the dynamics of a 12-state Quadrotor Model based on initial states

    :param x: 12 state input vector
    :type x: numpy.array
    :param t: The time
    :type t: float
    :param g: The gravity constant, defaults to 9.81
    :type g: float, optional
    :param R: The radius of center mass, defaults to 0.1
    :type R: float, optional
    :param l: The distance of motors to center mass, defaults to 0.5
    :type l: float, optional
    :param M_rotor: The motor mass, defaults to 0.1
    :type M_rotor: float, optional
    :param M: The center mass, defaults to 1.0
    :type M: float, optional
    :param u1: The desired height, defaults to 1.0
    :type u1: float, optional
    :param u2: The desired roll, defaults to 0.0
    :type u2: float, optional
    :param u3: The desired pitch, defaults to 0.0
    :type u3: float, optional
    :return: The partial derivatives of the 12-state quadrotor model
    :rtype: list
    """
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = x

    # total mass
    m = M + 4.0 * M_rotor

    # moments of inertia
    Jx = 2.0 / 5.0 * M * (R ** 2) + 2.0 * (l ** 2) * M_rotor
    Jy = Jx
    Jz = 2.0 / 5.0 * M * (R ** 2) + 4.0 * (l ** 2) * M_rotor

    # height control
    F = (m * g - 10.0 * (x3 - u1)) + 3.0 * x6

    # roll control
    tau_phi = -(x7 - u2) - x10

    # pitch control
    tau_theta = -(x8 - u3) - x11

    # heading uncontrolled
    tau_psi = 0

    dx1 = (
        np.cos(x8) * np.cos(x9) * x4
        + (np.sin(x7) * np.sin(x8) * np.cos(x9) - np.cos(x7) * np.sin(x9)) * x5
        + (np.cos(x7) * np.sin(x8) * np.cos(x9) + np.sin(x7) * np.sin(x9)) * x6
    )
    dx2 = (
        np.cos(x8) * np.sin(x9) * x4
        + (np.sin(x7) * np.sin(x8) * np.sin(x9) + np.cos(x7) * np.cos(x9)) * x5
        + (np.cos(x7) * np.sin(x8) * np.sin(x9) - np.sin(x7) * np.cos(x9)) * x6
    )
    dx3 = np.sin(x8) * x4 - np.sin(x7) * np.cos(x8) * x5 - np.cos(x7) * np.cos(x8) * x6
    dx4 = x12 * x5 - x11 * x6 - g * np.sin(x8)
    dx5 = x10 * x6 - x12 * x4 + g * np.cos(x8) * np.sin(x7)
    dx6 = x11 * x4 - x10 * x5 + g * np.cos(x8) * np.cos(x7) - F / m
    dx7 = x10 + np.sin(x7) * np.tan(x8) * x11 + np.cos(x7) * np.tan(x8) * x12
    dx8 = np.cos(x7) * x11 - np.sin(x7) * x12
    dx9 = np.sin(x7) / np.cos(x8) * x11 + np.cos(x7) / np.cos(x8) * x12
    dx10 = (Jy - Jz) / Jx * x11 * x12 + 1 / Jx * tau_phi
    dx11 = (Jz - Jx) / Jy * x10 * x12 + 1 / Jy * tau_theta
    dx12 = (Jx - Jy) / Jz * x10 * x11 + 1 / Jz * tau_psi

    dx = [dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8, dx9, dx10, dx11, dx12]

    return dx


def laub_loomis(x, t):
    """Defines the dynamics of the Laub-Loomis model based on initial states

    :param x: 7 state input vector
    :type x: numpy.array
    :param t: The time
    :type t: float
    :return: The partial derivatives of the 7-state Laub-Loomis model
    :rtype: list
    """
    x1, x2, x3, x4, x5, x6, x7 = x

    dx1 = 1.4 * x3 - 0.9 * x1
    dx2 = 2.5 * x5 - 1.5 * x2
    dx3 = 0.6 * x7 - 0.8 * x2 * x3
    dx4 = 2 - 1.3 * x3 * x4
    dx5 = 0.7 * x1 - x4 * x5
    dx6 = 0.3 * x1 - 3.1 * x6
    dx7 = 1.8 * x6 - 1.5 * x2 * x7

    dx = [dx1, dx2, dx3, dx4, dx5, dx6, dx7]

    return dx
