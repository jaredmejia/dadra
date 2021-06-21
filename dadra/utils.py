import cvxpy as cp
import numpy as np

from functools import partial


def solve_p_norm(n_x, sample, p=2, const=None):
    """
    Solve the scenario relaxation problem for the given sample with p-Norm Balls.

    :param n_x: The state dimension.
    :type n_x: int

    :param sample: Sample from dynamical system (num_samples, n_x).
    :type sample: numpy.ndarray

    :param p: The order of p-norm.
    :type p: int

    :param const: The constraints placed on the parameters A and b
    :type const: string
    """
    if const is None:
        A = cp.Variable((n_x, n_x), symmetric=True)
        b = cp.Variable((n_x, 1))
    elif const == "diagonal":
        a = cp.Variable((n_x, 1))
        A = cp.diag(a)
        b = cp.Variable((n_x, 1))
    elif const == "scalar":
        sigma = cp.Variable()
        A = sigma * np.identity(n_x)
        b = np.zeros((3, 1))

    obj = cp.Minimize(-cp.log_det(A))
    constraints = [cp.pnorm(A @ r.reshape(3, 1) - b, p=p) <= 1 for r in sample]
    prob = cp.Problem(obj, constraints)
    prob.solve()

    if const != "scalar":
        return A.value, b.value, prob.status
    else:
        return A, b, prob.status


def p_norm_cont(arr, axis, default_val, n_x, A_val, b_val, p, minimum=True):
    """
    Solve for the optimal value that satisfies the p-Norm Ball conditions
    at the specified axis.

    :param arr: Array of shape (n_x - 1,) containing the independent variables
    of the p-norm condition.
    :type arr: numpy.ndarray

    :param axis: The axis of the dependent variable for which to solve for
    (i.e. z -> axis=2).
    :type axis: int

    :param default_val: The value to return if no solution for the dependent variable
    is found that satisfies the p-norm conditions.
    :type default_val: float

    :parm n_x: The state dimension.
    :type n_x: int

    :param A_val: The matrix of shape (n_x, n_x) corresponding to the optimal p-norm ball.
    :type A_val: numpy.ndarray

    :param b_val: The vector of shape (n_x, 1) corresponding to the optimal p-norm ball.
    :type b_val: numpy.ndarray

    :param p: The order of p-norm.
    :type p: int

    :param minimum: True if optimizing for the minimal value of the dependent variable
    that satisfies the p-norm conditions.
    :type minimum: bool
    """
    vec = cp.Variable((n_x, 1))
    other_dims = list(range(n_x))
    other_dims.remove(axis)
    constraints = [vec[i][0] == arr[j] for i, j in zip(other_dims, range(n_x - 1))]
    constraints.append(cp.pnorm(A_val @ vec - b_val, p=p) <= 1)
    if minimum:
        obj = cp.Minimize(vec[axis])
    else:
        obj = cp.Maximize(vec[axis])
    prob = cp.Problem(obj, constraints)

    try:
        prob.solve()
    except:
        return default_val

    if prob.status != "optimal":
        return default_val

    return vec.value[axis]


def compute_contour(
    sample, A_val, b_val, cont_axis=2, n_x=3, p=2, grid_n=200, minimum=True
):
    """
    Computes the contour for 3 dimensions based on sample data and the A_val,
    and b_val corresponding to the optimal p-norm ball.

    :param sample: Sample from dynamical system (num_samples, n_x).
    :type sample: numpy.ndarray

    :param A_val: The matrix of shape (n_x, n_x) corresponding to the optimal p-norm ball.
    :type A_val: numpy.ndarray

    :param b_val: The vector of shape (n_x, 1) corresponding to the optimal p-norm ball.
    :type b_val: numpy.ndarray

    :param cont_axis: The axis for which the contours are to be solved for.
    :type cont_axis: int

    :param n_x: The state dimension.
    :type: n_x: int

    :param p: The order of p-norm.
    :type p: int

    :param grid_n: The side length of the cube of points to be used for computing
    contours.
    :type grid_n: int

    :param minimum: True if optimizing for the minimal value of the dependent variable
    that satisfies the p-norm conditions.
    :type minimum: bool
    """
    x_min, x_max = sample[:, 0].min(), sample[:, 0].max()
    y_min, y_max = sample[:, 1].min(), sample[:, 1].max()
    z_min, z_max = sample[:, 2].min(), sample[:, 2].max()

    x = np.linspace(
        x_min - 0.4 * (x_max - x_min), x_max + 0.4 * (x_max - x_min), grid_n
    )
    y = np.linspace(
        y_min - 0.2 * (y_max - y_min), y_max + 0.4 * (y_max - y_min), grid_n
    )
    z = np.linspace(
        x_min - 0.2 * (z_max - z_min), z_max + 0.4 * (z_max - z_min), grid_n
    )

    if cont_axis == 2:
        d0, d1 = np.meshgrid(x, y)
        c_min, c_max = z_min, z_max
    elif cont_axis == 1:
        d0, d1 = np.meshgrid(x, z)
        c_min, c_max = y_min, y_max
    elif cont_axis == 0:
        d0, d1 = np.meshgrid(y, z)
        c_min, c_max = x_min, x_max

    d2 = np.array([d0.flatten(), d1.flatten()]).T

    if minimum:
        default = z_max + 1
    else:
        default = z_min - 1

    solve_cont_d2 = partial(
        p_norm_cont,
        axis=cont_axis,
        default_val=default,
        n_x=n_x,
        A_val=A_val,
        b_val=b_val,
        p=p,
        minimum=minimum,
    )
    cont = np.fromiter(map(solve_cont_d2, d2), dtype=np.float64).reshape(grid_n, grid_n)

    return d0, d1, cont, c_min, c_max


def empirical_estimate(samples, A_val, b_val, n_x=3, p=2):
    """
    Computes the ratio of samples within the estimated reachability set.

    :param sample: Sample from dynamical system (num_samples, n_x).
    :type sample: numpy.ndarray

    :param A_val: The matrix of shape (n_x, n_x) corresponding to the optimal p-norm ball.
    :type A_val: numpy.ndarray

    :param b_val: The vector of shape (n_x, 1) corresponding to the optimal p-norm ball.
    :type b_val: numpy.ndarray

    :param n_x: The state dimension.
    :type: n_x: int

    :param p: The order of p-norm.
    :type p: int

    """
    num_samples = samples.shape[0]
    count = 0
    for sample in samples:
        vec = sample.reshape(n_x, 1)
        if np.linalg.norm(A_val @ vec - b_val, ord=p) <= 1:
            count += 1
    return count / num_samples
