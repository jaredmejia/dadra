Examples
=============

Installation/Usage:
*******************
DaDRA is published on PyPi and can be installed with ``pip install --upgrade dadra``.

For the most up to date version of DaDRA, the suggested method of installation is to clone the repository from git: ``git clone https://github.com/JaredMejia/dadra.git``.

For more information on the requirements and dependencies of the library, please see `this page <https://github.com/JaredMejia/dadra/blob/main/requirements.txt>`_

Once installed, the library and all its modules can be imported simply by calling ``import dadra``. 


Lorenz System Example
**************************************************
.. code-block:: python

    import dadra 

    # define the dynamics of the system
    def lorenz_system(z, t, sigma=10.0, rho=28.0, beta=8 / 3):
        dzdt = [
            sigma * (z[1] - z[0]),
            z[0] * (rho - z[2]) - z[1],
            z[0] * z[1] - beta * z[2],
        ]
        return dzdt

    # define the intervals for the initial states of the variables in the system
    state_dim = 3
    intervals = [(0, 1) for i in range(state_dim)]

    # instantiate a DynamicSystem object
    lorenz_ds = dadra.DynamicSystem(lorenz_system, intervals, state_dim)

    # instantiate an Estimator object
    lorenz_e = dadra.Estimator(
        dyn_sys=lorenz_ds, epsilon=0.05, delta=1e-9, p=2, const=None, normalize=True
    )
    """
    Upon instantiation of an Estimator object, the number of samples required to 
    satisfy the requested probabilistic guarantees is computed, and that number of
    samples are drawn from the dynamic system. Also, the p-norm ball that estimates
    the reachable set is computed.
    """

    # print a summary of the Estimator instance and attributes
    lorenz_e.summary()
    """
    ----------------------------------------------------------------
    Estimator Summary
    ================================================================
    State dimension: 3
    Accuracy parameter epsilon: 0.05
    Confidence parameter delta: 1e-09
    p-norm p value: 2
    Constraints on p-norm ball: None
    Number of samples: 941
    Status of p-norm ball solution: optimal
    ----------------------------------------------------------------
    """

    # save a plot of the samples to "figure_1.png"
    lorenz_e.plot_samples("figure_1.png")

    # save a plot of the 2D contours of the estimated reachable set to "figure_2.png"
    lorenz_e.plot_2D_cont(200, "figure_2.png")

    # save a plot of the 3D contours of the estimated reachable set to "figure_3.png"
    # save a rotating gif of the 3D contours of the estimated reachable set to "figure_4.gif"
    lorenz_e.plot_3D_cont(100, "figure_3.png", gif_name="figure_4.gif")
    