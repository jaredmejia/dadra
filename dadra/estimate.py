import numpy as np
import time
import warnings

from dadra.dyn_sys import Sampler, SimpleSystem, System
from dadra.utils.graph_utils import (
    grow_plot_3d,
    plot_contour_2D,
    plot_contour_3D,
    plot_contour_3D_time,
    plot_reach_time,
    plot_reach_time_2D,
    plot_reach_time_3D,
    plot_sample,
    plot_sample_time,
)
from dadra.utils.christoffel_utils import (
    c_compute_contours,
    c_emp_estimate,
    c_num_samples,
    construct_inv_christoffel,
    construct_kernelized_inv_christoffel,
)
from dadra.utils.misc_utils import format_time
from dadra.utils.p_utils import (
    multi_p_norm,
    p_compute_contour_2D,
    p_compute_contour_3D,
    p_compute_vals,
    p_dict_list,
    p_emp_estimate,
    p_get_dict,
    p_get_reachable_2D,
    p_get_reachable_3D,
    p_num_samples,
    solve_p_norm,
)
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm.auto import trange


class Estimator:
    """Class which allows for estimation of the reachable sets for a given dynamical system.

    :param dyn_sys: An instance of :class:`dadra.System`
    :type dyn_sys: System
    :param epsilon: The accuracy parameter, defaults to 0.05
    :type epsilon: float, optional
    :param delta: The confidence parameter, defaults to 1e-9
    :type delta: float, optional
    :param christoffel: If True, uses Christoffel functions to estimate the reachable set. If False, the p-norm method is used, defaults to False
    :type christoffel: bool, optional
    :param p: The order of p-norm, defaults to 2
    :type p: int, optional
    :param const: The constraints placed on the parameters A and b, defaults to None
    :type const: str, optional
    :param normalize: If true, the sample is normalized, defaults to True
    :type normalize: bool, optional
    :param d: The degree of polynomial features points are mapped to, defaults to 10
    :type d: int, optional
    :param kernelized: If True, the kernelized inverse christoffel function is constructed, defaults to False
    :type kernelized: bool, optional
    :param rho: Inverse Christoffel function constant rho, defaults to 0.0001
    :type rho: float, optional
    :param rbf: If True, a radical basis function kernel is used, defaults to False
    :type rbf: bool, optional
    :param scale: The length scale of the kernel, defaults to 1.0
    :type scale: float, optional
    :param iso_dim: Isolates the samples at the specified dimensions, currently only implemented when dyn_sys.all_time is True, defaults to None
    :type iso_dim: list, optional
    """

    def __init__(
        self,
        dyn_sys: System,
        epsilon=0.05,
        delta=1e-9,
        christoffel=False,
        p=2,
        const=None,
        normalize=True,
        d=10,
        kernelized=False,
        rho=0.0001,
        rbf=False,
        scale=1.0,
        iso_dim=None,
    ):

        self.dyn_sys = dyn_sys
        self.check_sys()

        self.epsilon = epsilon
        self.delta = delta
        self.christoffel = christoffel

        if not christoffel:
            self.p = p
            self.const = const
        else:
            self.const = const
            self.d = d
            self.kernelized = kernelized
            self.rho = rho
            self.rbf = rbf
            self.scale = scale

        self.normalize = normalize
        self.num_samples = self.get_num_samples()

        self.samples = None
        self.iso_dim = None

        if not christoffel:
            if not self.dyn_sys.all_time:
                self.A = None
                self.b = None
                self.status = None
            else:
                self.iso_dim = (
                    iso_dim
                    if isinstance(iso_dim, list) or iso_dim is None
                    else [iso_dim]
                )
                self.num_iso_samples = self.get_num_samples(iso=True)
                self.solution_list = None
                self.num_opt = None
                self.num_opt_in = None
                self.num_fail = None
        else:
            self.C = None
            self.level = None

        if iso_dim is not None:
            self.iso_samples = None

    @classmethod
    def estimator_from_sample_func(
        cls,
        sample_fn,
        state_dim,
        timesteps,
        parts,
        all_time=False,
        epsilon=0.05,
        delta=1e-9,
        christoffel=False,
        p=2,
        const=None,
        normalize=True,
        d=10,
        kernelized=False,
        rho=0.0001,
        rbf=False,
        scale=0.5,
        iso_dim=None,
    ):
        """Class method that allows for an instance of :class:`dadra.Estimator` to be initialized using a sampling function, and the components for an instance of :class:`Sampler` rather than explicitly passing in a :class:`System` object

        :param sample_fn: A function to sample from
        :type sample_fn: function
        :param state_dim: The degrees of freedom of the system
        :type state_dim: int
        :param timesteps: The number of timesteps over which to compute the sample, defaults to 100
        :type timesteps: int, optional
        :param parts: The number of parts to partition the time interval into for computing the sample, defaults to 1001
        :type parts: int, optional
        :param all_time: If True, each sample will include all timesteps from the system, rather than only the last timestep, defaults to False
        :type all_time: bool, optional
        :param epsilon: The accuracy parameter, defaults to 0.05
        :type epsilon: float, optional
        :param delta: The confidence parameter, defaults to 1e-9
        :type delta: float, optional
        :param christoffel: If True, uses Christoffel functions to estimate the reachable set. If False, the p-norm method is used, defaults to False
        :type christoffel: bool, optional
        :param p: The order of p-norm, defaults to 2
        :type p: int, optional
        :param const: The constraints placed on the parameters A and b, defaults to None
        :type const: string, optional
        :param normalize: If true, the sample is normalized, defaults to True
        :type normalize: bool, optional
        :param d: The degree of polynomial features points are mapped to, defaults to 10
        :type d: int, optional
        :param kernelized: If True, the kernelized inverse christoffel function is constructed, defaults to False
        :type kernelized: bool, optional
        :param rho: Inverse Christoffel function constant rho, defaults to 0.0001
        :type rho: float, optional
        :param rbf: If True, a radical basis function kernel is used, defaults to False
        :type rbf: bool, optional
        :param scale: The length scale of the kernel, defaults to 1.0
        :type scale: float, optional
        :return: A :class:`dadra.Estimator` object
        :rtype: :class:`dadra.Estimator`
        :param iso_dim: Isolates the samples at the specified dimensions, currently only implemented when dyn_sys.all_time is True, defaults to None
        :type iso_dim: list, optional
        :return: A :class:`dadra.Estimator` object
        :rtype: :class:`dadra.Estimator`
        """
        sampler = Sampler(sample_fn, state_dim, timesteps, parts, all_time=all_time)
        return cls(
            sampler,
            epsilon=epsilon,
            delta=delta,
            christoffel=christoffel,
            p=p,
            const=const,
            normalize=normalize,
            d=d,
            kernelized=kernelized,
            rho=rho,
            rbf=rbf,
            scale=scale,
            iso_dim=iso_dim,
        )

    @classmethod
    def estimator_from_func(
        cls,
        dyn_func,
        intervals,
        state_dim,
        timesteps=100,
        parts=1001,
        epsilon=0.05,
        delta=1e-9,
        christoffel=False,
        p=2,
        const=None,
        normalize=True,
        d=10,
        kernelized=False,
        rho=0.0001,
        rbf=False,
        scale=0.5,
        iso_dim=None,
    ):
        """Class method that allows for an instance of :class:`dadra.Estimator` to be initialized
        using a dynamic function, and the components for an instance of :class:`SimpleSystem`
        rather than explicitly passing in a :class:`System` object

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
        :param epsilon: The accuracy parameter, defaults to 0.05
        :type epsilon: float, optional
        :param delta: The confidence parameter, defaults to 1e-9
        :type delta: float, optional
        :param christoffel: If True, uses Christoffel functions to estimate the reachable set. If False, the p-norm method is used, defaults to False
        :type christoffel: bool, optional
        :param p: The order of p-norm, defaults to 2
        :type p: int, optional
        :param const: The constraints placed on the parameters A and b, defaults to None
        :type const: string, optional
        :param normalize: If true, the sample is normalized, defaults to True
        :type normalize: bool, optional
        :param d: The degree of polynomial features points are mapped to, defaults to 10
        :type d: int, optional
        :param kernelized: If True, the kernelized inverse christoffel function is constructed, defaults to False
        :type kernelized: bool, optional
        :param rho: Inverse Christoffel function constant rho, defaults to 0.0001
        :type rho: float, optional
        :param rbf: If True, a radical basis function kernel is used, defaults to False
        :type rbf: bool, optional
        :param scale: The length scale of the kernel, defaults to 1.0
        :type scale: float, optional
        :return: A :class:`dadra.Estimator` object
        :rtype: :class:`dadra.Estimator`
        :param iso_dim: Isolates the samples at the specified dimensions, currently only implemented when dyn_sys.all_time is True, defaults to None
        :type iso_dim: list, optional
        :return: A :class:`dadra.Estimator` object
        :rtype: :class:`dadra.Estimator`
        """
        dyn_sys = SimpleSystem(dyn_func, intervals, state_dim, timesteps, parts)
        return cls(
            dyn_sys,
            epsilon=epsilon,
            delta=delta,
            christoffel=christoffel,
            p=p,
            const=const,
            normalize=normalize,
            d=d,
            kernelized=kernelized,
            rho=rho,
            rbf=rbf,
            scale=scale,
            iso_dim=iso_dim,
        )

    @classmethod
    def estimator_from_list(
        cls,
        func_list,
        intervals,
        timesteps=100,
        parts=1001,
        epsilon=0.05,
        delta=1e-9,
        christoffel=False,
        p=2,
        const=None,
        normalize=True,
        d=10,
        kernelized=False,
        rho=0.0001,
        rbf=False,
        scale=0.5,
        iso_dim=None,
    ):
        """Class method that allows for an instance of :class:`dadra.Estimator` to be initialized
        using a list of dynamic functions, and the components for an instance of :class:`SimpleSystem`
        rather than explicitly passing in a :class:`System` object

        :param dyn_func: The list of functions, one for each variable, that define the dynamics of the system
        :type dyn_func: function
        :param intervals: The intervals corresponding to the possible values of the initial states of the variables in the system
        :type intervals: list
        :param timesteps: The number of timesteps over which to compute the sample, defaults to 100
        :type timesteps: int, optional
        :param parts: The number of parts to partition the time interval into for computing the sample, defaults to 1001
        :type parts: int, optional
        :param epsilon: The accuracy parameter, defaults to 0.05
        :type epsilon: float, optional
        :param delta: The confidence parameter, defaults to 1e-9
        :type delta: float, optional
        :param christoffel: If True, uses Christoffel functions to estimate the reachable set. If False, the p-norm method is used, defaults to False
        :type christoffel: bool, optional
        :param p: The order of p-norm, defaults to 2
        :type p: int, optional
        :param const: The constraints placed on the parameters A and b, defaults to None
        :type const: string, optional
        :param normalize: If true, the sample is normalized, defaults to True
        :type normalize: bool, optional
        :param d: The degree of polynomial features points are mapped to, defaults to 10
        :type d: int, optional
        :param kernelized: If True, the kernelized inverse christoffel function is constructed, defaults to False
        :type kernelized: bool, optional
        :param rho: Inverse Christoffel function constant rho, defaults to 0.0001
        :type rho: float, optional
        :param rbf: If True, a radical basis function kernel is used, defaults to False
        :type rbf: bool, optional
        :param scale: The length scale of the kernel, defaults to 1.0
        :type scale: float, optional
        :param iso_dim: Isolates the samples at the specified dimensions, currently only implemented when dyn_sys.all_time is True, defaults to None
        :type iso_dim: list, optional
        :return: A :class:`dadra.Estimator` object
        :rtype: :class:`dadra.Estimator`
        """
        dyn_sys = SimpleSystem.get_system(func_list, intervals, timesteps, parts)
        return cls(
            dyn_sys,
            epsilon=epsilon,
            delta=delta,
            christoffel=christoffel,
            p=p,
            const=const,
            normalize=normalize,
            d=d,
            kernelized=kernelized,
            rho=rho,
            rbf=rbf,
            scale=scale,
            iso_dim=iso_dim,
        )

    def check_sys(self):
        """Checks whether the dynamic system passed to the constructor is valid

        :raises TypeError: If an object of type other than :class:`dadra.System` is passed
        """
        if not isinstance(self.dyn_sys, System):
            raise TypeError("Object of type System must be passed")

    def get_num_samples(self, iso=False):
        """Compute the number of samples needed to satisfy the specified probabilistic guarantees

        :return: The number of samples needed to satisfy the specified probabilistic guarantees
        :rtype: int
        """
        if not iso:
            n_x = self.dyn_sys.state_dim
        else:
            n_x = len(self.iso_dim)
        if not self.christoffel:
            return p_num_samples(self.epsilon, self.delta, n_x, self.const)
        else:
            return c_num_samples(self.epsilon, self.delta, n_x, self.d)

    def get_sample(self):
        """Draws the number of samples necessary to satisfy the specified probabilistic guarantees

        :return: Array of samples
        :rtype: numpy.ndarray
        """
        print(f"Drawing {self.num_samples} samples")
        start_sampling = time.perf_counter()
        samples = self.dyn_sys.sample_system(self.num_samples)
        end_sampling = time.perf_counter()

        print(
            f"Time to draw {self.num_samples} samples: {format_time(end_sampling - start_sampling)}"
        )

        if self.normalize:
            samples = (samples - np.mean(samples, axis=0)) / np.std(samples)

        return samples

    def estimate(self):
        """Both draws the samples and makes an estimate of the reachable set"""
        self.sample_system()
        self.compute_estimate()

    def sample_system(self):
        """Draws and stores samples from the system"""
        self.samples = self.get_sample()
        print(f"Shape of all samples: {self.samples.shape}")

        if self.iso_dim is not None:
            self.iso_samples = self.samples[: self.num_iso_samples, :, self.iso_dim]
            print(f"Shape of dimension reduced samples: {self.iso_samples.shape}")

    def set_iso_dim(self, iso_dim):
        """Sets the iso_dim attribute and updates iso_samples attribute to correspond to the new iso_dim

        :param iso_dim: Isolates the samples at the specified dimensions
        :type iso_dim: list, optional
        """
        self.iso_dim = (
            iso_dim if isinstance(iso_dim, list) or iso_dim is None else [iso_dim]
        )
        if self.iso_dim is not None:
            self.num_iso_samples = self.get_num_samples(iso=True)
            print(f"Number of samples for isolated dimensions: {self.num_iso_samples}")
            self.iso_samples = self.samples[: self.num_iso_samples, :, self.iso_dim]

    def compute_estimate(self):
        """Computes an estimate of the reachable set based on the type of method specified"""
        if not self.christoffel:
            if not self.dyn_sys.all_time:
                self.A, self.b, self.status = self.solve_p()
            else:
                self.solution_list = self.solve_p()
        else:
            self.C, self.level = self.construct_christoffel()

    def solve_p(self):
        """Solves for the optimal p-norm ball that estimates the reachable set

        :return: The matrix corresponding to parameter A of the p-norm ball, the vector corresponding to parameter b of the p-norm ball, and the status of the optimization problem
        :rtype: tuple
        """
        print(f"Solving for optimal p-norm ball (p={self.p})")

        if self.iso_dim is None:
            curr_samples = self.samples
        else:
            print(
                f"Solving for p-norm ball with respect to time at isolated dimensions: {self.iso_dim}"
            )
            curr_samples = self.iso_samples
            if len(curr_samples.shape) != 3:
                curr_samples = np.expand_dims(curr_samples, axis=2)

        start_time = time.perf_counter()
        if not self.dyn_sys.all_time:
            A, b, status = solve_p_norm(
                curr_samples, self.dyn_sys.state_dim, self.p, self.const
            )
            end_time = time.perf_counter()

            print(
                f"Time to solve for optimal p-norm ball: {format_time(end_time - start_time)}"
            )

            if status != "optimal":
                warnings.warn(
                    f"Solution for p-norm ball is not optimal: status = {status}"
                )

            return A, b, status

        else:
            solution_list = multi_p_norm(curr_samples, self.p, self.const)
            end_time = time.perf_counter()

            print(
                f"Time to solve for optimal p-norm ball for all timesteps: {format_time(end_time - start_time)}"
            )

            start_check = time.perf_counter()
            solution_list = self.check_statuses(solution_list)
            end_check = time.perf_counter()

            print(
                f"Time to check each solution status: {format_time(end_check-start_check)}"
            )

            if self.num_opt != len(solution_list):
                warnings.warn(
                    "Not all p-norm ball solutions across the timesteps are optimal accurate"
                    + "\n"
                    + f"number of optimal solutions: {self.num_opt}"
                    + "\n"
                    + f"number of optimal inaccurate solutions: {self.num_opt_in}"
                    + "\n"
                    + f"number of infeasible or unbounded solutions: {self.num_fail}"
                )

            return solution_list

    def check_statuses(self, solution_list=None):
        """Checks the status of each of the solutions in the list of p-norm ball solutions

        :param solution_list: List of solutions where each solution is a tuple of the form (A, b, status), defaults to None
        :type solution_list: list, optional
        """
        self.num_opt = 0
        self.num_opt_in = 0
        self.num_fail = 0
        s_list = solution_list if solution_list is not None else self.solution_list
        failed_indices = []
        for i in range(len(s_list)):
            status = s_list[i]["status"]
            if status == "optimal":
                self.num_opt += 1
            elif status == "optimal_inaccurate":
                self.num_opt_in += 1
            else:
                self.num_fail += 1
                failed_indices.append(i)
        print(f"{failed_indices=}")
        s_list = [j for k, j in enumerate(s_list) if k not in failed_indices]
        if self.iso_samples is not None:
            self.iso_samples = np.delete(self.iso_samples, failed_indices, axis=1)
        else:
            self.samples = np.delete(self.samples, failed_indices, axis=1)
        return s_list

    def construct_christoffel(self):
        """Constructs the inverse Christoffel function and the respective level parameter from the samples

        :return: the inverse Christoffel function and the level parameter
        :rtype: tuple
        """
        if self.kernelized:
            C = construct_kernelized_inv_christoffel(
                self.samples, self.d, rho=self.rho, rbf=self.rbf, scale=self.scale
            )
        else:
            C = construct_inv_christoffel(self.samples, self.d)

        start_lvl_param = time.perf_counter()
        values = C(self.samples)
        level = np.max(values)
        end_lvl_param = time.perf_counter()
        print(
            f"Time to compute level parameter: {format_time(end_lvl_param - start_lvl_param)}"
        )
        return C, level

    def empirical_estimate(self, num_samples_emp=None):
        """Computes the ratio of samples within the estimated reachable set

        :param num_samples_emp: The number of samples to draw for the empirical estimate, defaults to the number of samples drawn to obtain the p-norm ball
        :type num_samples_emp: int, optional
        """
        if not self.christoffel and self.status != "optimal":
            warnings.warn(
                "Non-optimal solution to p-norm ball. Empirical estimate may be innaccurate"
            )

        if num_samples_emp is None:
            num_samples_emp = self.num_samples

        print(f"Drawing {num_samples_emp} samples for empirical estimate")
        start_sampling = time.perf_counter()
        samples_emp = self.dyn_sys.sample_system(num_samples_emp)
        end_sampling = time.perf_counter()

        print(
            f"Time to draw {num_samples_emp} samples: {format_time(end_sampling - start_sampling)}"
        )

        if self.normalize:
            samples_emp = (samples_emp - np.mean(samples_emp, axis=0)) / np.std(
                samples_emp
            )

        if not self.christoffel:
            ratio = p_emp_estimate(
                samples_emp, self.A, self.b, self.dyn_sys.state_dim, self.p
            )
        else:
            ratio = c_emp_estimate(self.samples, self.C, self.level)

        print(f"Ratio of samples within the estimated reachable set: {ratio}")

    def summary(self):
        """Prints a summary of the attributes of this instance of :class:`Estimator`"""
        summary_str = (
            "--------------------------------------------------------------------"
            + "\n"
        )
        summary_str += "Estimator Summary" + "\n"
        summary_str += (
            "===================================================================="
            + "\n"
        )
        summary_str += f"State dimension: {self.dyn_sys.state_dim}" + "\n"
        summary_str += f"Accuracy parameter epsilon: {self.epsilon}" + "\n"
        summary_str += f"Confidence parameter delta: {self.delta}" + "\n"
        summary_str += f"Number of samples: {self.num_samples}" + "\n"
        if not self.christoffel:
            summary_str += f"Method of estimation: p-Norm Ball" + "\n"
            summary_str += f"p-norm p value: {self.p}" + "\n"
            summary_str += f"Constraints on p-norm ball: {self.const}" + "\n"
            if not self.dyn_sys.all_time:
                if all([self.A is None, self.b is None, self.status is None]):
                    summary_str += (
                        f"Status of p-norm ball solution: No estimate has been made yet"
                        + "\n"
                    )
                else:

                    summary_str += (
                        f"Status of p-norm ball solution: {self.status}" + "\n"
                    )
            else:
                if self.iso_dim is not None:
                    summary_str += f"Isolated dimensions: {self.iso_dim}" + "\n"
                    summary_str += (
                        f"Number of samples for isolated dimensions: {self.num_iso_samples}"
                        + "\n"
                    )
                if self.solution_list is None:
                    summary_str += (
                        f"Status of p-norm ball solutions: No estimates have been made yet"
                        + "\n"
                    )
                else:
                    summary_str += (
                        f"Ratio of failed p-norm ball solutions: {self.num_fail / len(self.solution_list)}"
                        + "\n"
                    )
        else:
            summary_str += f"Method of estimation: Inverse Christoffel Function" + "\n"
            summary_str += f"Degree of polynomial features: {self.d}" + "\n"
            summary_str += f"Kernelized: {self.kernelized}" + "\n"
            summary_str += f"Constant rho: {self.rho}" + "\n"
            summary_str += f"Radical basis function kernel: {self.rbf}" + "\n"
            summary_str += f"Length scale of kernel: {self.scale}" + "\n"
            if all([self.C is None, self.level is None]):
                summary_str += (
                    f"Status of Christoffel Function estimate: No estimate has been made yet"
                    + "\n"
                )
            else:
                summary_str += f"Christoffel function level: {self.level}" + "\n"
                summary_str += (
                    f"Status of Christoffel function estimate: Estimate made" + "\n"
                )

        summary_str += (
            "--------------------------------------------------------------------"
            + "\n"
        )
        print(summary_str)

    def plot_samples(
        self,
        fig_name,
        gif_name=None,
        num_samples_show=100,
        figsize=(10, 10),
        color="default",
        **kwargs,
    ):
        """Plots the samples of 2-dimensional shape in 2D or samples of 3-dimensional shape in 3-dimensions drawn with respect to time

        :param fig_name: The name of the file to save the plot to
        :type fig_name: str
        :param gif_name: The name of the animated gif to be saved, only applicable for 3-dimensional plots when all_time is True, defaults to None
        :type gif_name: str, optional
        :param num_samples_show: The maximum number of samples to plot, defaults to 100
        :type num_samples_show: int, optional
        :param figsize: the size of the figure to be saved, defaults to (10, 10)
        :type figsize: tuple, optional
        :param color: The color to draw the lines in, defaults to "default"
        :type color: str, optional
        :raises ValueError: If the samples are of an incompatible shape
        """
        if not self.dyn_sys.all_time and self.iso_dim is None:
            plot_start = time.perf_counter()
            plot_sample(self.samples[:num_samples_show], fig_name)
            plot_end = time.perf_counter()
            print(f"Time to plot samples: {format_time(plot_end - plot_start)}")
        else:
            if len(self.iso_samples.shape) == 2 or self.iso_samples.shape[2] == 1:
                plot_start = time.perf_counter()
                if len(self.iso_samples.shape) != 2:
                    sample_squeezed = np.squeeze(self.iso_samples, axis=2)
                else:
                    sample_squeezed = self.iso_samples
                time_x = np.linspace(0, self.dyn_sys.timesteps, self.dyn_sys.parts)

                plot_sample_time(
                    time_x,
                    sample_squeezed[:num_samples_show],
                    fig_name,
                    figsize=figsize,
                    color=color,
                    **kwargs,
                )
                plot_end = time.perf_counter()
                print(
                    f"Time to plot samples in 2D over time: {format_time(plot_end - plot_start)}"
                )
            elif len(self.iso_samples.shape) == 3 and self.iso_samples.shape[2] == 2:
                plot_start = time.perf_counter()
                plot_sample_time(
                    None,
                    self.iso_samples,
                    fig_name,
                    figsize=figsize,
                    color=color,
                    **kwargs,
                )
                plot_end = time.perf_counter()
                print(
                    f"Time to plot samples in 2D over time: {format_time(plot_end - plot_start)}"
                )
            elif len(self.iso_samples.shape) == 3 and self.iso_samples.shape[2] == 3:
                plot_start = time.perf_counter()
                grow_plot_3d(
                    self.iso_samples[:num_samples_show],
                    fig_name,
                    gif_name,
                    figsize,
                    **kwargs,
                )
                plot_end = time.perf_counter()
                print(
                    f"Time to plot samples in 3D over time: {format_time(plot_end - plot_start)}"
                )

            else:
                raise ValueError(
                    f"Cannot handle samples of shape: {self.iso_samples.shape}"
                )

    def get_reachable_3D(self, i, grid_n=25):
        if self.solution_list is None:
            raise ValueError("Reachable set estimate has not been computed yet")

        curr_samples = (
            self.iso_samples if self.iso_samples is not None else self.samples
        )
        curr_solution = self.solution_list[i]

        return p_get_reachable_3D(
            curr_samples[:, i, :],
            curr_solution["A"],
            curr_solution["b"],
            grid_n=grid_n,
        )

    def plot_reachable_time(
        self,
        fig_name,
        gif_name=None,
        grid_n=200,
        num_samples_show=50,
        figsize=(10, 10),
        **kwargs,
    ):
        """Plots the reachable set estimate (as well as possibly some samples) over time. Currently only implemented for p-norm balls

        :param fig_name: The name of the file to save the plot to
        :type fig_name: str
        :param gif_name: The name of the animated gif to be saved, only applicable for 3-dimensional plots, defaults to None
        :type gif_name: str, optional
        :param grid_n: The number of points to test for the p-norm ball estimation at each time step, defaults to 200
        :type grid_n: int, optional
        :param num_samples_show: The number of sample trajectories to show in addition to the reachable set, defaults to 50
        :type num_samples_show: int, optional
        :param figsize: the size of the figure to be saved, defaults to (10, 10)
        :type figsize: tuple, optional
        """
        if not self.dyn_sys.all_time:
            warnings.warn(
                "`plot_reachable_time` is meant for samples across all timesteps"
            )

        if self.iso_dim is None:
            curr_samples = self.samples
        else:
            curr_samples = self.iso_samples

        if len(curr_samples.shape) == 2 or curr_samples.shape[2] == 1:
            if len(curr_samples.shape) != 2:
                curr_samples = np.squeeze(curr_samples, axis=2)

        if len(curr_samples.shape) == 2:
            plot_start = time.perf_counter()
            min_vals, max_vals = [], []
            for i in range(len(self.solution_list)):
                A_i, b_i = self.solution_list[i]["A"], self.solution_list[i]["b"]
                vals_i = p_compute_vals(curr_samples[:, i], A_i, b_i, grid_n=grid_n)
                try:
                    min_vals.append(min(vals_i))
                    max_vals.append(max(vals_i))
                except ValueError:
                    print(f"{i=}")

            time_x = np.linspace(0, self.dyn_sys.timesteps, len(self.solution_list))

            plot_reach_time(
                time_x,
                curr_samples,
                min_vals,
                max_vals,
                fig_name,
                num_samples_show=num_samples_show,
                figsize=figsize,
                **kwargs,
            )
            plot_end = time.perf_counter()
            print(
                f"time to plot 2D samples with respect to time: {format_time(plot_end - plot_start)}"
            )

        elif len(curr_samples.shape) == 3 and curr_samples.shape[2] == 2:
            plot_start = time.perf_counter()
            all_p_norm = []
            for i in range(len(self.solution_list)):
                xs, ys = p_get_reachable_2D(
                    curr_samples[:, i, :],
                    self.solution_list[i]["A"],
                    self.solution_list[i]["b"],
                    p=2,
                    grid_n=grid_n,
                )
                all_p_norm.append((xs, ys))

            plot_reach_time_2D(
                curr_samples[:num_samples_show], all_p_norm, fig_name, **kwargs
            )
            plot_end = time.perf_counter()
            print(
                f"time to plot 2D samples with respect to time: {format_time(plot_end - plot_start)}"
            )

        elif len(curr_samples.shape) == 3 and curr_samples.shape[2] == 3:
            p = Pool(cpu_count())
            start_solve = time.perf_counter()
            tuple_list_3D = p.map(
                self.get_reachable_3D, trange(len(self.solution_list))
            )
            end_solve = time.perf_counter()
            print(f"time to get reachable set: {format_time(end_solve - start_solve)}")
            plot_start = time.perf_counter()
            plot_reach_time_3D(
                curr_samples[:num_samples_show],
                tuple_list_3D,
                fig_name,
                gif_name=gif_name,
                **kwargs,
            )
            # dict_list = p_dict_list(
            #     curr_samples,
            #     self.solution_list,
            #     self.dyn_sys.parts,
            #     num_indices=20,
            #     logspace=True,
            #     grid_n=grid_n,
            # )
            # plot_contour_3D_time(
            #     curr_samples[:num_samples_show],
            #     dict_list,
            #     fig_name,
            #     gif_name,
            #     figsize,
            #     **kwargs,
            # )
            plot_end = time.perf_counter()
            print(
                f"time to plot 3D samples with respect to time: {format_time(plot_end - plot_start)}"
            )

        else:
            raise ValueError(
                f"Cannot handle samples of shape: {self.iso_samples.shape}"
            )

    def plot_2D_cont(self, fig_name, grid_n=200):
        """Computes the contours of the reachable set and plots them in 2D

        :param fig_name: The name of the file to save the plot to
        :type fig_name: str
        :param grid_n: The side length of the cube of points to be used for computing contours, defaults to 200
        :type grid_n: int
        """
        if not self.christoffel:
            cont_compute = partial(
                p_compute_contour_2D,
                sample=self.samples,
                A_val=self.A,
                b_val=self.b,
                n_x=self.dyn_sys.state_dim,
                p=self.p,
                grid_n=grid_n,
            )

            print("Computing 2D contours")
            cont_start = time.perf_counter()
            xv1, yv1, z_cont, z_min, z_max = cont_compute(cont_axis=2)
            xv2, zv1, y_cont, y_min, y_max = cont_compute(cont_axis=1)
            yv2, zv2, x_cont, x_min, x_max = cont_compute(cont_axis=0)
            cont_end = time.perf_counter()
            print(f"Time to compute contours: {format_time(cont_end - cont_start)}")

            plot_start = time.perf_counter()
            plot_contour_2D(
                xv1,
                yv1,
                z_cont,
                z_max,
                self.samples,
                fig_name,
                xv2=xv2,
                zv1=zv1,
                y_cont=y_cont,
                y_level=y_max,
                yv2=yv2,
                zv2=zv2,
                x_cont=x_cont,
                x_level=x_max,
            )
            plot_end = time.perf_counter()
            print(f"time to plot contours in 2D: {format_time(plot_end - plot_start)}")

        else:
            start_comp_cont = time.perf_counter()
            xv, yv, cont = c_compute_contours(self.C, self.samples, grid_n)
            end_comp_cont = time.perf_counter()
            print(
                f"Time to compute contour: {format_time(end_comp_cont - start_comp_cont)}"
            )
            plot_start = time.perf_counter()
            plot_contour_2D(xv, yv, cont, self.level, self.samples, fig_name)
            plot_end = time.perf_counter()
            print(f"time to plot contours in 2D: {format_time(plot_end - plot_start)}")

    def plot_3D_cont(self, fig_name, grid_n=100, gif_name=None):
        """Computes and plots the contours in 3D with the option for saving an animated gif of the rotating graph

        :param fig_name: The name of the file to save the plot to
        :type fig_name: str
        :param grid_n: The side length of the cube of points to be used for computing contours, defaults to 100
        :type grid_n: int
        :param gif_name: The name of the file to save the gif to, defaults to None
        :type gif_name: str, optional
        """
        print("Computing 3D contour")
        cont_start = time.perf_counter()
        d0, d1, cont_min, cont_max, c_min, c_max = p_compute_contour_3D(
            self.samples, self.A, self.b, grid_n=grid_n
        )
        cont_end = time.perf_counter()
        print(f"Time to compute contours: {format_time(cont_end - cont_start)}")

        plot_start = time.perf_counter()
        plot_contour_3D(
            d0,
            d1,
            cont_min,
            c_min,
            c_max,
            self.samples,
            fig_name,
            gif_name=gif_name,
            z_cont2=cont_max,
        )
        plot_end = time.perf_counter()
        print(f"time to plot contours in 3D: {format_time(plot_end - plot_start)}")
