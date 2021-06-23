import numpy as np

from collections.abc import Callable
from functools import partial
from inspect import signature
from numpy.random import default_rng
from sampling import make_sample_n
from scipy.integrate import odeint


class System:
    def __init__(self, dynamics, intervals, timesteps=100, parts=1001):
        if len(dynamics) > len(intervals):
            raise ValueError(
                "There must be a function and interval defined for each variable"
            )

        if type(timesteps) != int:
            raise ValueError("The number of timesteps must be an integer")

        self.dynamics = dynamics
        self.intervals = intervals
        self.timesteps = timesteps
        self.parts = parts
        self.state_dim = len(self.dynamics)

        self.check_dyn_sig()
        self.check_intervals()

    def check_dyn_sig(self):
        for func in self.dynamics:
            if not isinstance(func, Callable):
                raise TypeError(
                    "There must be a callable function defined for each variable"
                )

            sig = signature(func)
            num_extra_intervs = len(self.intervals) - self.state_dim
            if num_extra_intervs == 0 and len(sig.parameters) != 2:
                raise ValueError(
                    "Each dynamic function must be a function of the initial state and time"
                )
            elif num_extra_intervs > 0 and len(sig.parameters) != 2 + (
                num_extra_intervs
            ):
                raise ValueError(
                    "Each dynamic function must be a function of the initial state, time, and the provided additional arguments"
                )

    def check_intervals(self):
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

    def dyn_to_sys(self, arr, t, *args):
        if len(args) != 0:
            derivs = [d(arr, t, *args) for d in self.dynamics]
        else:
            derivs = [d(arr, t) for d in self.dynamics]
        return derivs

    def system_sampler(self, x=None):
        t = np.linspace(0, self.timesteps, self.parts)
        ru = default_rng().uniform
        rand_intervals = [ru(lower, upper) for lower, upper in self.intervals]
        initial_state = np.array(rand_intervals[: self.state_dim])
        extra_intervs = rand_intervals[self.state_dim :]
        if len(extra_intervs) != 0:
            sol = odeint(self.dyn_to_sys, initial_state, t, args=tuple(extra_intervs))
        else:
            sol = odeint(self.dyn_to_sys, initial_state, t)
        return sol[-1]

    def sample_system(self, N):
        return make_sample_n(self.system_sampler)(N)
