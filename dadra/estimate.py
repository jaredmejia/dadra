from functools import partial
import numpy as np
import time
import warnings

from dadra.dyn_sys import SimpleSystem, System
from dadra.utils.graph_utils import plot_contour_2D, plot_contour_3D, plot_sample
from dadra.utils.christoffel_utils import (
    c_compute_contours,
    c_emp_estimate,
    c_num_samples,
    construct_inv_christoffel,
    construct_kernelized_inv_christoffel,
)
from dadra.utils.misc_utils import format_time
from dadra.utils.p_utils import (
    p_compute_contour_2D,
    p_compute_contour_3D,
    p_emp_estimate,
    p_num_samples,
    solve_p_norm,
)


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

        if not christoffel:
            self.A = None
            self.b = None
            self.status = None
        else:
            self.C = None
            self.level = None

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
        )

    def check_sys(self):
        """Checks whether the dynamic system passed to the constructor is valid

        :raises TypeError: If an object of type other than :class:`dadra.System` is passed
        """
        if not isinstance(self.dyn_sys, System):
            raise TypeError("Object of type System must be passed")

    def get_num_samples(self):
        """Compute the number of samples needed to satisfy the specified probabilistic guarantees

        :return: The number of samples needed to satisfy the specified probabilistic guarantees
        :rtype: int
        """
        n_x = self.dyn_sys.state_dim
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
        """Draws the samples and makes an estimate of the reachable set"""
        self.samples = self.get_sample()

        if not self.christoffel:
            self.A, self.b, self.status = self.solve_p()
        else:
            self.C, self.level = self.construct_christoffel()

    def solve_p(self):
        """Solves for the optimal p-norm ball that estimates the reachable set

        :return: The matrix corresponding to parameter A of the p-norm ball, the vector corresponding to parameter b of the p-norm ball, and the status of the optimization problem
        :rtype: tuple
        """
        print(f"Solving for optimal p-norm ball (p={self.p})")
        start_time = time.perf_counter()
        A, b, status = solve_p_norm(
            self.samples, self.dyn_sys.state_dim, self.p, self.const
        )
        end_time = time.perf_counter()
        if status != "optimal":
            warnings.warn("Failed to find optimal solution for p-norm ball")
        else:
            print(
                f"Time to solve for optimal p-norm ball: {format_time(end_time - start_time)}"
            )
        return A, b, status

    def construct_christoffel(self):
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
            "----------------------------------------------------------------" + "\n"
        )
        summary_str += "Estimator Summary" + "\n"
        summary_str += (
            "================================================================" + "\n"
        )
        summary_str += f"State dimension: {self.dyn_sys.state_dim}" + "\n"
        summary_str += f"Accuracy parameter epsilon: {self.epsilon}" + "\n"
        summary_str += f"Confidence parameter delta: {self.delta}" + "\n"
        summary_str += f"Number of samples: {self.num_samples}" + "\n"
        if not self.christoffel:
            summary_str += f"Method of estimation: p-Norm Ball"
            summary_str += f"p-norm p value: {self.p}" + "\n"
            summary_str += f"Constraints on p-norm ball: {self.const}" + "\n"
            if all([self.A is None, self.b is None, self.status is None]):
                summary_str += (
                    f"Status of p-norm ball solution: No estimate has been made yet"
                    + "\n"
                )
            else:
                summary_str += f"Status of p-norm ball solution: {self.status}" + "\n"
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
                summary_str += (
                    f"Status of Christoffel Function estimate: Estimate made" + "\n"
                )

        summary_str += (
            "----------------------------------------------------------------" + "\n"
        )
        print(summary_str)

    def plot_samples(self, fig_name):
        """Plots the samples of of shape (num_items, 3) in 3D

        :param fig_name: The name of the file to save the plot to
        :type fig_name: string
        """
        plot_sample(self.samples, fig_name)

    def plot_2D_cont(self, fig_name, grid_n=200):
        """Computes the contours of the reachable set and plots them in 2D

        :param fig_name: The name of the file to save the plot to
        :type fig_name: string
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
            print(f"Time to compute 2D contours: {format_time(cont_end - cont_start)}")

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

        else:
            start_comp_cont = time.perf_counter()
            xv, yv, cont = c_compute_contours(self.C, self.samples, grid_n)
            end_comp_cont = time.perf_counter()
            print(
                f"Time to compute contour: {format_time(end_comp_cont - start_comp_cont)}"
            )

            plot_contour_2D(xv, yv, cont, self.level, self.samples, fig_name)

    def plot_3D_cont(self, fig_name, grid_n=100, gif_name=None):
        """Computes and plots the contours in 3D with the option for saving an animated gif of the rotating graph

        :param fig_name: The name of the file to save the plot to
        :type fig_name: string
        :param grid_n: The side length of the cube of points to be used for computing contours, defaults to 100
        :type grid_n: int
        :param gif_name: The name of the file to save the gif to, defaults to None
        :type gif_name: string, optional
        """
        print("Computing 3D contour")
        cont_start = time.perf_counter()
        d0, d1, cont_min, cont_max, c_min, c_max = p_compute_contour_3D(
            self.samples, self.A, self.b, grid_n=grid_n
        )
        cont_end = time.perf_counter()
        print(f"Time to compute 3D contour: {format_time(cont_end - cont_start)}")

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
