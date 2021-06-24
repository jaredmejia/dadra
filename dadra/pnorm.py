import numpy as np

from collections.abc import Callable
from functools import partial
from inspect import signature, Parameter
from numpy.random import default_rng, uniform
from sampling import make_sample_n
from scipy.integrate import odeint


class DynamicSystem:
    def __init__(self, dyn_func, intervals, state_dim, timesteps=100, parts=1001):
        self.dyn_func = dyn_func
        self.intervals = intervals
        self.state_dim = state_dim
        self.timesteps = timesteps
        self.parts = parts

        self.check_intervals()
        self.check_dyn_func()

    @classmethod
    def get_system(cls, func_list, intervals, timesteps=100, parts=1001):
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
        return make_sample_n(self.system_sampler)(N)
