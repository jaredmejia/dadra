import math
import numpy as np
import time

from dadra.utils.misc_utils import format_time, normalize, normalize0
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import PolynomialFeatures


def c_num_samples(epsilon, delta, n_x=2, k=10):
    """Compute the number of samples needed to satisfy the specified probabilistic guarantees for the inverse Christoffel function reachable set estimate

    :param epsilon: The accuracy parameter
    :type epsilon: float
    :param delta: The confidence parameter
    :type delta: float
    :param n_x: The state dimension, defaults to 2
    :type n_x: int
    :param k: degree of polynomial features points are mapped to
    :type k: int
    :return: The number of samples needed to satisfy the specified probabilistic guarantees
    :rtype: int
    """
    return math.ceil(
        (5 / epsilon)
        * (math.log(4 / delta) + math.comb(n_x + 2 * k, n_x) * math.log(40 / epsilon))
    )


def construct_inv_christoffel(data, d):
    """Constructs the inverse christoffel function from data

    :param data: A R^{n x p} matrix, n input points of dimension p
    :type data: numpy.ndarray
    :param d: degree of polynomial features points are mapped to
    :type d: int
    :return: C, the degree d inverse christoffel function constructed from data
    :rtype: function
    """
    # [a, b] -> [1, a, b, a^2, ab, b^2]
    poly_map = PolynomialFeatures(degree=d).fit_transform
    time1 = time.perf_counter()
    poly_mapped_data = poly_map(data)
    time2 = time.perf_counter()
    print(f"Time to apply polynomial mapping to data: {format_time(time2-time1)}")

    # scale V for numerical stability
    V = normalize0(poly_mapped_data).T
    # V is z_k(x), vector of monomials degree <= k (d)
    # M is matrix of moments
    M = V @ V.T
    time3 = time.perf_counter()
    print(f"Time to construct moment matrix: {format_time(time3-time2)}")

    # Moore-Penrose psuedo-inverse of matrix
    M_inv = np.linalg.pinv(M)
    time4 = time.perf_counter()
    print(f"Time to (pseudo)invert moment matrix: {format_time(time4-time3)}")

    def C(x):
        """The empirical inverse christoffel function for a point cloud x of i.i.d. samples

        :param x: Point cloud of i.i.d. samples, shape R^p or R^{n' x p}
        :type x: numpy.ndarray
        :return: R^{n'} vector of evaluations on data
        :rtype: numpy.ndarray
        """
        z = poly_map(x)
        if len(z.shape) == 1:
            return z @ M_inv @ z
        elif len(z.shape) == 2:
            return ((z @ M_inv) * z).sum(axis=1)

    return C


def make_rbf_kernel(scale=1.0):
    """Constructs a radical basis function kernel

    :param scale: The length scale of the kernel, defaults to 1.0
    :type scale: float
    :return: The radical basis function kernel and the diagonal of the kernel
    :rtype: tuple
    """
    rbf = RBF(scale)

    def rbf_ker(x1, x2):
        """Computes the rbf kernel from two arrays

        :param x1: Left argument of the rbf kernel, array of shape (n_samples_x1, n_features)
        :type x1: numpy.ndarray
        :param x2: Right argument of the rbf kernel, array of shape (n_samples_x2, n_features)
        :type x2: numpy.ndarray
        :return: The rbf kernel k(x1, x2), array of shape (n_sample_x1, n_samples_x2)
        :rtype: numpy.ndarray
        """
        if len(x1.shape) == 1:
            x1 = x1[None]
        if len(x2.shape) == 1:
            x2 = x2[None]
        return np.squeeze(rbf(x1, x2))

    def diag_rbf_ker(x):
        """Computes the diagonal of the kernel

        :param x: The rbf kernel
        :type x: numpy.ndarray
        :return: The diagonal of the rbf kernel
        :rtype: numpy.ndarray
        """
        return rbf.diag(x)

    return rbf_ker, diag_rbf_ker


def construct_kernelized_inv_christoffel(data, d, rho=0.0001, rbf=False, scale=1.0):
    """Constructs the kernelized inverse christoffel function from data

    :param data: R^{n x p} matrix, n input points of dimension p
    :type data: numpy.ndarray
    :param d: degree of polynomial features points are mapped to
    :type d: int
    :param rho: Constant rho, defaults to 0.0001
    :type rho: float, optional
    :param rbf: If True, a radical basis function kernel is used, defaults to False
    :type rbf: bool, optional
    :param scale: The length scale of the kernel, defaults to 1.0
    :type scale: int, optional
    :return: C, the kernelized degree d inverse christoffel function constructed from data
    :rtype: function
    """
    n = data.shape[0]
    if rbf:
        ker, diag_ker = make_rbf_kernel(scale)
    else:
        ker = lambda x1, x2: (1 + x1 @ x2.T) ** d
        diag_ker = lambda x: (1 + (x * x).sum(axis=1)) ** d

    time1 = time.perf_counter()
    X = normalize(data, 0, 1)
    VTV = ker(X, X)
    time2 = time.perf_counter()
    print(f"Time to construct data kernel matrix: {format_time(time2-time1)}")

    rho = rho * np.linalg.norm(VTV) / np.sqrt(n)
    K_inv = np.linalg.pinv(rho * np.eye(n) + VTV)
    time2 = time.perf_counter()
    print(f"Time to invert rho*I + kernel: {format_time(time2-time1)}")

    def C(x):
        """The empirical inverse christoffel function for a point cloud x of i.i.d. samples

        :param x: Point cloud of i.i.d. samples, shape R^p or R^{n' x p}
        :type x: numpy.ndarray
        :return: R^{n'} vector of evaluations on data
        :rtype: numpy.ndarray
        """
        x = normalize(x, 0, 1)
        if len(x.shape) == 1:
            kappa = ker(x, x)
            VTv = ker(X, x)
            return (kappa - VTv @ K_inv @ VTv) / rho
        elif len(x.shape) == 2:
            kappa = diag_ker(x)
            VTv = ker(X, x)
            return (kappa - (VTv * (K_inv @ VTv)).sum(axis=0)) / rho

    return C


def c_compute_contours(C, samples, grid_n=200):
    """Computes contours of C in region surrounding the input samples

    :param C: The inverse Christoffel function
    :type C: function
    :param samples: Samples from dynamical system (num_samples, n_x)
    :type samples: numpy.ndarray
    :param grid_n: The side length of the cube of points to be used for computing contours, defaults to 200
    :type grid_n: int, optional
    :return: The meshgrid, and the corresponding computed contour
    :rtype: tuple
    """
    x_min = samples[:, 0].min()
    x_max = samples[:, 0].max()
    y_min = samples[:, 1].min()
    y_max = samples[:, 1].max()
    x = np.linspace(
        x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min), grid_n
    )
    y = np.linspace(
        y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min), grid_n
    )
    xv, yv = np.meshgrid(x, y)
    z = np.array([xv.flatten(), yv.flatten()]).T
    cont = C(z)
    return xv, yv, cont.reshape(grid_n, grid_n)


def c_emp_estimate(samples, C, level):
    """Computes the ratio of samples within the estimated reachable set for the Christoffel function reachable set estimation

    :param samples: Sample from dynamical system (num_samples, n_x)
    :type samples: numpy.ndarray
    :param C: The inverse Christoffel function
    :type C: function
    :param level: The maximum level value of the Christoffel function on the original sample
    :type level: float
    :return: The ratio of samples within the estimated reachability set
    :rtype: float
    """
    values = C(samples)
    num_elem = np.prod(samples.shape)
    ratio = 1 - np.where(values > level, 1, 0).sum() / num_elem
    return ratio
