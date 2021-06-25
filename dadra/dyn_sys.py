import numpy as np

from collections.abc import Callable
from inspect import signature, Parameter
from numpy.random import default_rng
from dadra.sampling import make_sample_n
from scipy.integrate import odeint


class DynamicSystem:
    """Class implementation of a dynamical system that allows for parallelized sampling.

    :param dyn_func: The function defining the dynamics of the system
    :type dyn_func: function
    :param intervals: The intervals corresponding to the possible values of the initial states of the variables in the system
    :type intervals: list
    :param state_dim: The degrees of freedom of the system
    :type state_dim: int
    :param timesteps: The number of timesteps over which to compute the sample, defaults to 100
    :type timesteps: int, optional
    :param parts: The number of parts to partition the time interval into for computing the sample, defaults to 1001
    :type parts: int, optional
    """

    def __init__(self, dyn_func, intervals, state_dim, timesteps=100, parts=1001):
        """Constructor method"""
        self.dyn_func = dyn_func
        self.intervals = intervals
        self.state_dim = state_dim
        self.timesteps = timesteps
        self.parts = parts

        self.check_intervals()
        self.check_dyn_func()

    @classmethod
    def get_system(cls, func_list, intervals, timesteps=100, parts=1001):
        """Class method that allows for an instance of :class:`dadra.DynamicSystem` to be initialized
        using a list of functions, one for each variable, to define the dynamics of a system rather than
        a single function

        :param func_list: The list of functions, one for each variable, that define the dynamics of the system
        :type func_list: list
        :param intervals: The intervals corresponding to the possible values of the initial states of the variables in the system
        :type intervals: list
        :param timesteps: The number of timesteps over which to compute the sample, defaults to 100
        :type timesteps: int, optional
        :param parts: The number of parts to partition the time interval into for computing the sample, defaults to 1001
        :type parts: int, optional
        :raises TypeError: If not all functions in ``func_list`` are callable
        :raises ValueError: If not all functions in ``func_list`` include parameters for the intial state and time
        :raises ValueError: If not all functions in ``func_list`` include parameters for the intial state, time, and the provided additional arguments
        :return: A :class:`dadra.DynamicSystem` object
        :rtype: :class:`dadra.DynamicSystem`
        """
        state_dim = len(func_list)
        num_intervs = len(intervals)
        num_extra_intervs = num_intervs - state_dim

        for func in func_list:
            if not isinstance(func, Callable):
                raise TypeError(
                    "There must be a callable function defined for each variable"
                )
            sig = signature(func)
            num_params = len(sig.parameters)

            if num_extra_intervs == 0 and num_params != 2:
                raise ValueError(
                    "Each dynamic function must be a function of the initial state and time"
                )
            elif num_extra_intervs > 0 and num_params != (2 + num_extra_intervs):
                raise ValueError(
                    "Each dynamic function must be a function of the initial state, time, and the provided additional arguments"
                )

        global dyn_to_sys  # note: can only run experiments sequentially, create new object each time

        def dyn_to_sys(arr, t, *args):
            if len(args) != 0:
                derivs = [f(arr, t, *args) for f in func_list]
            else:
                derivs = [f(arr, t) for f in func_list]
            return derivs

        dyn_func = dyn_to_sys

        dyn_sys = cls(dyn_func, intervals, state_dim, timesteps, parts)
        return dyn_sys

    def check_intervals(self):
        """Checks whether the list of intervals in this instance of the :class:`dadra.DynamicSystem` class
        is valid

        :raises ValueError: If there is not an interval defined for each variable
        :raises ValueError: If not all intervals include an upper and a lower bound
        :raises ValueError: If not all intervals consist of two numbers
        """
        if len(self.intervals) < self.state_dim:
            raise ValueError("There must be an interval defined for each variable")

        for interval in self.intervals:
            if len(interval) != 2:
                raise ValueError(
                    "Each interval must include an upper and a lower bound"
                )
            for v in interval:
                if type(v) == int or type(v) == float:
                    pass
                else:
                    raise ValueError("Each interval must consist of two numbers")

    def check_dyn_func(self):
        """Checks whether the dynamic function in this instance of the :class:`dadra.DynamicSystem` class
        is valid

        :raises TypeError: If the provided dynamic function is not callable
        :raises ValueError: If there is an invalid number of required arguments in the dynamic function
        """
        if not isinstance(self.dyn_func, Callable):
            raise TypeError("A callable dynamic function must be provided")

        sig = signature(self.dyn_func)
        num_params = len(sig.parameters)
        num_required = num_params

        for p in sig.parameters.values():
            if p.default != Parameter.empty or p.kind == Parameter.VAR_POSITIONAL:
                num_required -= 1

        num_extra_intervs = len(self.intervals) - self.state_dim
        if num_required > (2 + num_extra_intervs):
            raise ValueError("Invalid number of required arguments in dynamic function")

    def system_sampler(self, x=None):
        """Obtains sample from specified system

        :param x: Placeholder variable over which to evaluate, defaults to None
        :type x: NoneType, optional
        :return: The sample from the specified system at the last timestep
        :rtype: numpy.ndarray
        """
        t = np.linspace(0, self.timesteps, self.parts)
        ru = default_rng().uniform
        rand_intervals = [ru(lower, upper) for lower, upper in self.intervals]
        initial_state = np.array(rand_intervals[: self.state_dim])
        extra_intervs = tuple(rand_intervals[self.state_dim :])
        if len(extra_intervs) != 0:
            sol = odeint(self.dyn_func, initial_state, t, args=extra_intervs)
        else:
            sol = odeint(self.dyn_func, initial_state, t)
        return sol[-1]

    def sample_system(self, N):
        """Draws ``N`` samples from the specified system

        :param N: The number of samples to be drawn
        :type N: int
        :return: Array of ``N`` samples
        :rtype: numpy.ndarray
        """
        return make_sample_n(self.system_sampler)(N)
