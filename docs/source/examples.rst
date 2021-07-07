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
    lorenz_e.plot_3D_cont(
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
