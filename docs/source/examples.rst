Installation/Usage:
===================

Installation:
*******************
DaDRA is published on PyPi and can be installed with ``pip install --upgrade dadra``.

For the most up to date version of DaDRA, the suggested method of installation is to clone the repository from git: ``git clone https://github.com/JaredMejia/dadra.git``.

For more information on the requirements and dependencies of the library, please see `this page <https://github.com/JaredMejia/dadra/blob/main/requirements.txt>`_

Once installed, the library and all its modules can be imported simply by calling ``import dadra``. 

Data-Driven Reachability Analysis Examples:
********************************************

Lorenz System with disturbance using Scenario Approach
--------------------------------------------------------
.. code-block:: python

    import dadra

    # define the dynamics of the system
    def lorenz_system(z, t, disturbance=None, eta=1.0, sigma=10.0, rho=28.0, beta=8 / 3):
        dzdt = [
            sigma * (z[1] - z[0]) + disturbance.get_dist(0, t) * eta,
            z[0] * (rho - z[2]) - z[1] + disturbance.get_dist(1, t) * eta,
            z[0] * z[1] - beta * z[2] + disturbance.get_dist(2, t) * eta,
        ]
        return dzdt

    # define the intervals for the initial states of the variables in the system
    l_state_dim = 3
    l_intervals = [(0, 1) for i in range(l_state_dim)]

    # instantiate a DisturbedSystem object for a disturbed system
    l_ds = dadra.DisturbedSystem(
        dyn_func=lorenz_system,
        intervals=l_intervals,
        state_dim=l_state_dim,
        disturbance=l_disturbance,
        timesteps=100,
        parts=1001,
    )

    # instantiate an Estimator object
    l_e = dadra.Estimator(dyn_sys=l_ds, epsilon=0.05, delta=1e-9, p=2, normalize=True)

    # print out a summary of the Estimator object
    l_e.summary()
    """
    ----------------------------------------------------------------
    Estimator Summary
    ================================================================
    State dimension: 3
    Accuracy parameter epsilon: 0.05
    Confidence parameter delta: 1e-09
    Number of samples: 941
    Method of estimation: p-Norm Ball
    p-norm p value: 2
    Constraints on p-norm ball: None
    Status of p-norm ball solution: No estimate has been made yet
    ----------------------------------------------------------------
    """

    # make a reachable set estimate on the disturbed system
    l_e.estimate()
    """
    Drawing 941 samples
    Using 16 CPUs

    100%
    941/941 [01:23<00:00, 15.12it/s]

    Time to draw 941 samples: 01 minutes and 24 seconds
    Solving for optimal p-norm ball (p=2)
    Time to solve for optimal p-norm ball: 00 minutes and 02 seconds
    """

    # print out a summary of the Estimator object once more
    """
    ----------------------------------------------------------------
    Estimator Summary
    ================================================================
    State dimension: 3
    Accuracy parameter epsilon: 0.05
    Confidence parameter delta: 1e-09
    Number of samples: 941
    Method of estimation: p-Norm Ball
    p-norm p value: 2
    Constraints on p-norm ball: None
    Status of p-norm ball solution: optimal
    ----------------------------------------------------------------
    """

    # save a plot of the 2D contours of the estimated reachable set
    l_e.plot_2D_cont("figures/l_estimate_2D.png", grid_n=200)
    """
    Computing 2D contours
    Time to compute 2D contours: 01 minutes and 05 seconds
    """

    # save a plot and a rotating gif of the 3D contours of the estimated reachable set
    l_e.plot_3D_cont(
        "figures/l_estimate_3D.png", grid_n=100, gif_name="figures/l_estimate_3D.gif"
    )


Duffing Oscillator using Christoffel Functions
-----------------------------------------------
.. code-block:: python

    import dadra
    import numpy as np

    # define the dynamics of the system
    def duffing_oscillator(y, t, alpha=0.05, omega=1.3, gamma=0.4):
        dydt = [y[1], -alpha * y[1] + y[0] - y[0] ** 3 + gamma * np.cos(omega * t)]
        return dydt

    # define the intervals for the initial states of the variables in the system
    d_state_dim = 2
    d_intervals = [(0.95, 1.05), (-0.05, 0.05)]

    # instantiate a SimpleSystem object for a non-disturbed system
    d_ds = dadra.SimpleSystem(
        dyn_func=duffing_oscillator,
        intervals=d_intervals,
        state_dim=d_state_dim,
        timesteps=100,
        parts=1001,
    )

    # instantiate an Estimator object
    d_e = dadra.Estimator(
        dyn_sys=d_ds,
        epsilon=0.05,
        delta=1e-9,
        christoffel=True,
        normalize=True,
        d=10,
        rho=0.0001,
        rbf=False,
        scale=1.0,
    )

    # print out a summary of the Estimator object
    d_e.summary()
    """
    -----------------------------------------------------------------------
    Estimator Summary
    =======================================================================
    State dimension: 2
    Accuracy parameter epsilon: 0.05
    Confidence parameter delta: 1e-09
    Number of samples: 156626
    Method of estimation: Inverse Christoffel Function
    Degree of polynomial features: 10
    Kernelized: False
    Constant rho: 0.0001
    Radical basis function kernel: False
    Length scale of kernel: 1.0
    Status of Christoffel function estimate: No estimate has been made yet
    -----------------------------------------------------------------------
    """

    # make a reachable set estimate on the disturbed system
    d_e.estimate()
    """
    Drawing 156626 samples
    Using 16 CPUs

    100%
    156626/156626 [03:46<00:00, 683.80it/s]

    Time to draw 156626 samples: 03 minutes and 47 seconds
    Time to apply polynomial mapping to data: 00 minutes and 00 seconds
    Time to construct moment matrix: 00 minutes and 00 seconds
    Time to (pseudo)invert moment matrix: 00 minutes and 00 seconds
    Time to compute level parameter: 00 minutes and 00 seconds
    """

    # print out a summary of the Estimator object once more
    d_e.summary()
    """
    -----------------------------------------------------------------------
    Estimator Summary
    =======================================================================
    State dimension: 2
    Accuracy parameter epsilon: 0.05
    Confidence parameter delta: 1e-09
    Number of samples: 156626
    Method of estimation: Inverse Christoffel Function
    Degree of polynomial features: 10
    Kernelized: False
    Constant rho: 0.0001
    Radical basis function kernel: False
    Length scale of kernel: 1.0
    Status of Christoffel function estimate: Estimate made
    -----------------------------------------------------------------------
    """

    # save a plot of the 2D contours of the estimated reachable set
    d_e.plot_2D_cont("figures/d_estimate_2D.png", grid_n=200)
    """
    Time to compute contour: 00 minutes and 00 seconds
    """

12 state Quadrotor using Scenario Approach
-----------------------------------------------
.. code-block:: python

    import dadra 
    import numpy as np

    # define the dynamics of the system
    def quadrotor(x, t, g=9.81, R=0.1, l=0.5, M_rotor=0.1, M=1.0, u1=1.0, u2=0.0, u3=0.0):
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

    # define the intervals for the initial states of the variables in the system
    q_state_dim = 12
    q_intervals = [(-0.4, 0.4) for _ in range(6)]
    q_intervals.extend([(0, 0) for _ in range(6)])

    # instantiate a SimpleSystem object for a non-disturbed system
    q_ss = dadra.SimpleSystem(
        dyn_func=quadrotor,
        intervals=q_intervals,
        state_dim=q_state_dim,
        timesteps=5,
        parts=100,
        all_time=True
    )

    # instantiate an Estimator object
    q_e = dadra.Estimator(
        dyn_sys=q_ss, 
        epsilon=0.05,
        delta=1e-9,
        p=2,
        normalize=False,
        iso_dim=2
    )
    """
    `iso_dim` specifies the dimension at which we will compute the reachable set over time.
    In this case, the `iso_dim=2` corresponds to the altitude of the quadrotor
    """

    # print out a summary of the Estimator object
    q_e.summary()
    """
    --------------------------------------------------------------------
    Estimator Summary
    ====================================================================
    State dimension: 12
    Accuracy parameter epsilon: 0.05
    Confidence parameter delta: 1e-09
    Number of samples: 3504
    Method of estimation: p-Norm Ball
    p-norm p value: 2
    Constraints on p-norm ball: None
    Status of p-norm ball solutions: No estimates have been made yet
    --------------------------------------------------------------------
    """

    # draw samples from the system
    q_e.sample_system()
    """
    Drawing 3504 samples
    Using 16 CPUs

    100%
    3504/3504 [00:05<00:00, 678.82it/s]

    Time to draw 3504 samples: 00 minutes and 05 seconds
    samples shape: (3504, 100, 1)
    """

    # compute a reachable set estimate
    q_e.compute_estimate()

    # print out a summary of the Estimator object once more
    q_e_4.summary()

    # plot the trajectories of the samples over time
    q_e_4.plot_samples("figures/quad_samples.png")
    """
    This saves a plot of altitude vs time 
    """

    # plot the reachable set estimate of the altitude over time 
    q_e.plot_reachable_time(
        "figures/quad_reachable.png",
        grid_n=200, 
        num_samples_show=50,
        x=[1],
        y=[0.9, 0.98, 1.02, 1.4]
    )