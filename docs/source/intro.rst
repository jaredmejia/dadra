Introduction
============

`DaDRA <https://github.com/JaredMejia/dadra>`_  (day-druh) is a Python library for Data-Driven Reachability Analysis. The main goal of the package is to accelerate the process of computing estimates of forward reachable sets for a given dynamical system.

Motivation
**********

Currently, there are no prominent Python libraries in use for Data-Driven Reachability Analysis, as most exist strictly in MATLAB. With the ever increasing popularity of the Python language, DaDRA takes advantage of the many open-source Python libraries already well established, such as ``cvxpy`` and ``scipy``, and applies them to the domain of control theory.

Limitations
***********

- As this library is still young, there is currently limited support for certain types of dynamical systems. Further improvements can be made to :class:`dadra.DynamicSystem` to expand the use and applicability of DaDRA.
- As of now, there only exists in DaDRA scenario reachability with p-norm balls. Other methods, including non-scenario approach methods such as Christoffel functions, will be implemented in the near future.
- To take full advantage of Python's ``multiprocessing.pool`` objects, the :class:`dadra.DynamicSystem` currently makes use of the ``global`` keyword upon the instantiation of an inner function within the :meth:`dadra.DynamicSystem.get_system` method. This allows pooling to be used for parallelization, but prevents multiple instances of :class:`dadra.DynamicSystem` to draw samples in a non-sequential order, though this is only a problem if each instance is instantiated using the class method :meth:`dadra.DynamicSystem.get_system`, rather than the default constructor for the :class:`dadra.DynamicSystem` class.