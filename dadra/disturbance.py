import numpy as np

from collections.abc import Callable, Iterable
from functools import partial
from numpy.random import default_rng
from sklearn.preprocessing import PolynomialFeatures


class ScalarDisturbance:
    """Class that models the scalar disturbance in a single dimension of a dynamic system

    :param basis_funcs: The functions factored into the scalar disturbance
    :type basis_funcs: list
    :param num_funcs: The number of basis functions
    :type num_funcs: int
    :param map_funcs: If true, `basis_funcs` is assumed to be a single function that plays the role of multiple basis functions when an input is passed to it, defaults to False
    :type map_funcs: bool, optional
    :param decay_weights: If True, the weights of the basis functions decrease as the index of the functions increase, defaults to False
    :type decay_weights: bool, optional
    :param negative_weights: If False, the weights of the functions must only be positive, defaults to True
    :type negative_weights: bool, optional
    """

    def __init__(
        self,
        basis_funcs,
        num_funcs,
        map_funcs=False,
        decay_weights=False,
        negative_weights=True,
    ):
        self.basis_funcs = basis_funcs
        self.num_funcs = num_funcs
        self.map_funcs = map_funcs
        self.decay_weights = decay_weights
        self.negative_weights = negative_weights

        self.check_funcs()
        self.ru = default_rng().uniform

        if not self.decay_weights:
            self.upper_alpha = 1
            if self.negative_weights:
                self.lower_alpha = -1
            else:
                self.lower_alpha = 0
        else:
            arr_upper = np.arange(1, self.num_funcs + 1)
            self.upper_alpha = 1 / arr_upper
            if self.negative_weights:
                self.lower_alpha = -self.upper_alpha
            else:
                self.lower_alpha = np.zeros(self.num_funcs)

        # by default functions are equally weighted
        self.alpha = np.ones(self.num_funcs)

    @classmethod
    def from_iterable(cls, basis_funcs, decay_weights=False, negative_weights=True):
        """Class method that allows for an instance of :class:`dadra.ScalarDisturbance` to be initialized directly from a list of functions

        :param basis_funcs: A list of basis functions to be factored into the scalar disturbance
        :type basis_funcs: list
        :param decay_weights: If True, the weights of the basis functions decrease as the index of the functions increase, defaults to False
        :type decay_weights: bool, optional
        :param negative_weights: If False, the weights of the functions must only be positive, defaults to True
        :type negative_weights: bool, optional
        :return: A :class:`dadra.ScalarDisturbance` object
        :rtype: :class:`dadra.ScalarDisturbance`
        """
        num_funcs = len(basis_funcs)
        return cls(
            basis_funcs,
            num_funcs,
            map_funcs=False,
            decay_weights=decay_weights,
            negative_weights=negative_weights,
        )

    @classmethod
    def sin_disturbance(cls, num_funcs, decay_weights=False, negative_weights=True):
        """Class method that allows for an instance of :class:`dadra.ScalarDisturbance` to be initialized directly from a list of sinusoidal basis functions

        :param num_funcs: The number of basis functions to include in the scalar disturbance
        :type num_funcs: int
        :param decay_weights: If True, the weights of the basis functions decrease as the index of the functions increase, defaults to False
        :type decay_weights: bool, optional
        :param negative_weights: If False, the weights of the functions must only be positive, defaults to True
        :type negative_weights: bool, optional
        :return: A :class:`dadra.ScalarDisturbance` object
        :rtype: :class:`dadra.ScalarDisturbance`
        """
        basis_funcs = [cls.sinusoid(k) for k in range(num_funcs)]
        return cls(
            basis_funcs,
            num_funcs,
            map_funcs=False,
            decay_weights=decay_weights,
            negative_weights=negative_weights,
        )

    @classmethod
    def poly_disturbance(cls, num_funcs, decay_weights=True, negative_weights=True):
        """Class method that allows for an instance of :class:`dadra.ScalarDisturbance` to be initialized directly from a list of polynomial basis functions

        :param num_funcs: The number of basis functions to include in the scalar disturbance
        :type num_funcs: int
        :param decay_weights: If True, the weights of the basis functions decrease as the index of the functions increase, defaults to False
        :type decay_weights: bool, optional
        :param negative_weights: If False, the weights of the functions must only be positive, defaults to True
        :type negative_weights: bool, optional
        :return: A :class:`dadra.ScalarDisturbance` object
        :rtype: :class:`dadra.ScalarDisturbance`
        """
        deg = num_funcs - 1
        basis_funcs = cls.poly(deg)
        return cls(
            basis_funcs,
            num_funcs,
            map_funcs=True,
            decay_weights=decay_weights,
            negative_weights=negative_weights,
        )

    @classmethod
    def no_disturbance(cls):
        """Class method that allows for an instance of :class:`dadra.ScalarDisturbance` to be initialized with no disturbance

        :return: A :class:`dadra.ScalarDisturbance` object
        :rtype: :class:`dadra.ScalarDisturbance`
        """
        basis_func = [cls.zero_func]
        return cls(basis_func, 1)

    @staticmethod
    def sinusoid(k):
        """Defines the k-th sinusoidal basis function of a scalar disturbance

        :param k: The index of the basis function
        :type k: int
        :return: The k-th sinusoidal basis function
        :rtype: function
        """
        if k == 0:
            return ScalarDisturbance.k_sin_one
        else:
            k_sin = partial(ScalarDisturbance.sin_basis, k=k)
            return k_sin

    @staticmethod
    def k_sin_one(t):
        """Constant function that returns 1

        :param t: The time
        :type t: float
        :return: 1
        :rtype: int
        """
        return 1

    @staticmethod
    def sin_basis(t, k):
        """The k-th sinusoidal basis function

        :param t: The time
        :type t: float
        :return: The output of the basis function at time t
        :rtype: float
        """
        return np.sin(2 * np.pi * k * t)

    @staticmethod
    def poly(deg):
        """Defines the mapping function which computes all basis functions for a polynomial disturbance

        :param deg: The maximum degree polynomial
        :type deg: int
        :return: The poylynomial mapping function
        :rtype: function
        """
        poly_map = PolynomialFeatures(degree=deg).fit_transform
        return poly_map

    @staticmethod
    def zero_func(t):
        """Defines the basis function for zero disturbance

        :param t: The time
        :type t: float
        :return: [description]
        :rtype: [type]
        """
        return 0

    def check_funcs(self):
        """Checks that the basis functions are all valid

        :raises TypeError: If 'basis_funcs' is not an Iterable when 'map_funcs' is set to False
        :raises TypeError: If the items within 'basis_funcs' are not callable functions when 'map_funcs' is set to False
        :raises TypeError: If 'basis_funcs' is not a callable function when 'map_funcs' is set to True
        """
        if not self.map_funcs:
            if not isinstance(self.basis_funcs, Iterable):
                raise TypeError(
                    "'ScalarDisturbance' object attribute 'basis_funcs' must be an Iterable if attribute 'map_funcs' is False"
                )
            for func in self.basis_funcs:
                if not isinstance(func, Callable):
                    raise TypeError(
                        "'ScalarDisturbance' object attribute 'basis_funcs' must consist of callable functions if attribute 'map_funcs' is False"
                    )
        else:
            if not isinstance(self.basis_funcs, Callable):
                raise TypeError(
                    "'ScalarDisturbance' object attribute 'basis_funcs' must be a callable function if attribute 'map_funcs' is True"
                )

    def draw_alpha(self):
        """Initializes alpha, the weights for the basis functions, from an m-dimensional random distribution"""
        self.alpha = self.ru(self.lower_alpha, self.upper_alpha, size=(self.num_funcs,))

    def d(self, t):
        """Computes the scalar disturbance at the specified time

        :param t: The time
        :type t: float
        :return: The scalar disturbance
        :rtype: float
        """
        if not self.map_funcs:
            weighted_funcs = [a * f(t) for a, f in zip(self.alpha, self.basis_funcs)]
            return sum(weighted_funcs)
        else:
            t_arr = np.array([[t]])
            y = self.basis_funcs(t_arr)
            return float(np.dot(y, self.alpha))


class Disturbance:
    """Class that models the disturbance for all n-dimensions of a dynamic system

    :param dist_list: A list of objects of type :class:`dadra.ScalarDisturbance`
    :type dist_list: list
    """

    def __init__(self, dist_list):
        self.dist_list = dist_list
        self.num_dists = len(dist_list)

    @classmethod
    def n_sin_disturbance(
        cls, n, num_funcs=5, decay_weights=False, negative_weights=True
    ):
        """Class method that creates an instance of :class:`dadra.Disturbance` consisting of sinusoidal disturbance for each dimension

        :param n: The dimension of the disturbance, corresponding to the state dimension of the dynamic system
        :type n: int
        :param num_funcs: The number of basis functions (m) for each scalar dimension, defaults to 5
        :type num_funcs: int, optional
        :param decay_weights: If True, the weights of the basis functions decrease as the index of the functions increase, defaults to False
        :type decay_weights: bool, optional
        :param negative_weights: If False, the weights of the basis functions must only be positive, defaults to True
        :type negative_weights: bool, optional
        :return: A :class:`dadra.Disturbance` object
        :rtype: :class:`dadra.Disturbance`
        """
        dist_list = [
            ScalarDisturbance.sin_disturbance(
                num_funcs,
                decay_weights=decay_weights,
                negative_weights=negative_weights,
            )
            for _ in range(n)
        ]
        return cls(dist_list)

    @classmethod
    def n_poly_disturbance(
        cls, n, num_funcs=5, decay_weights=True, negative_weights=True
    ):
        """Class method that creates an instance of :class:`dadra.Disturbance` consisting of polynomial disturbance for each dimension

        :param n: The dimension of the disturbance, corresponding to the state dimension of the dynamic system
        :type n: int
        :param num_funcs: The number of basis functions (m) for each scalar dimension, defaults to 5
        :type num_funcs: int, optional
        :param decay_weights: If True, the weights of the basis functions decrease as the index of the functions increase, defaults to False
        :type decay_weights: bool, optional
        :param negative_weights: If False, the weights of the basis functions must only be positive, defaults to True
        :type negative_weights: bool, optional
        :return: A :class:`dadra.Disturbance` object
        :rtype: :class:`dadra.Disturbance`
        """
        dist_list = [
            ScalarDisturbance.poly_disturbance(
                num_funcs,
                decay_weights=decay_weights,
                negative_weights=negative_weights,
            )
            for _ in range(n)
        ]
        return cls(dist_list)

    @classmethod
    def n_zero_disturbance(cls, n):
        """Class method that creates an instance of :class:`dadra.Disturbance` consisting of zero disturbance for each dimension

        :param n: The dimension of the disturbance, corresponding to the state dimension of the dynamic system
        :type n: int
        :return: A :class:`dadra.Disturbance` object
        :rtype: :class:`dadra.Disturbance`
        """
        dist_list = [ScalarDisturbance.no_disturbance() for _ in range(n)]
        return cls(dist_list)

    def draw_alphas(self):
        """Initialize the alphas, the weights for the basis functions, for each separate scalar disturbance"""
        for dist in self.dist_list:
            dist.draw_alpha()

    def get_dist(self, n, t):
        """Computes the scalar disturbance at the specified dimension and time

        :param n: The dimension of the scalar disturbance
        :type n: int
        :param t: The time
        :type t: float
        :return: The scalar disturbance
        :rtype: float
        """
        return self.dist_list[n].d(t)
