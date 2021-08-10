<img src="https://github.com/JaredMejia/dadra/blob/1b84928cc6672fb5beb2d98fb6572342cc6d713c/imgs/big_dadra_logo.png?raw=true" width="400">

# Python Library for Data-Driven Reachability Analysis
[![Documentation Status](https://readthedocs.org/projects/dadra/badge/?version=latest)](https://dadra.readthedocs.io/en/latest/?badge=latest)
[![PyPI Latest Release](https://img.shields.io/pypi/v/dadra.svg)](https://pypi.org/project/dadra/)
[![Conda Latest Release](https://anaconda.org/jaredmejia/dadra/badges/version.svg)](https://anaconda.org/jaredmejia/dadra)
[![License](https://img.shields.io/pypi/l/dadra.svg)](https://github.com/dadra/blob/master/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p float="left">
  <img src="https://github.com/JaredMejia/dadra/blob/1b84928cc6672fb5beb2d98fb6572342cc6d713c/imgs/l_3D_cont.gif?raw=true" width="300" height="300">
  <img src="https://github.com/JaredMejia/dadra/blob/1b84928cc6672fb5beb2d98fb6572342cc6d713c/imgs/quad_reachable.png?raw=true" width="300" height="300">
</p>

## What is it?
**DaDRA** (day-druh) is a Python library for Data-Driven Reachability Analysis. The main goal of the package is to accelerate the process of computing estimates of forward reachable sets for nonlinear dynamical systems. For more information about the library, see [this poster from the NSF funded SUPERB Program](https://github.com/JaredMejia/dadra/blob/34042b23cda31c6b0ba7c8acfe7f379ea4c6c434/imgs/SUPERB_DADRA_poster.png)

## Installation
To install the current release of DaDRA:
```
$ pip install --upgrade dadra
```
or
```
$ conda install -c jaredmejia dadra
```

## Resources
* [PyPi](https://pypi.org/project/dadra/)
* [Anaconda](https://anaconda.org/jaredmejia/dadra)
* [Documentation](https://dadra.readthedocs.io/en/latest/)
* [Issue tracking](https://github.com/JaredMejia/dadra/issues)

## Usage
See these examples from the [documentation](https://dadra.readthedocs.io/en/latest/):
* [Lorenz System with disturbance using Scenario Approach](https://dadra.readthedocs.io/en/latest/examples.html#lorenz-system-with-disturbance-using-scenario-approach)
* [Duffing Oscillator using Christoffel Functions](https://dadra.readthedocs.io/en/latest/examples.html#duffing-oscillator-using-christoffel-functions)
* [12-state Quadrotor using Scenario Approach](https://dadra.readthedocs.io/en/latest/examples.html#state-quadrotor-using-scenario-approach)

## Contributing
For contributions, please follow the workflow:
  1. **Fork** the repo on GitHub
  2. **Clone** the project to your own machine
  3. **Commit** changes to your own branch
  4. **Push** your work back up to your fork
  5. Submit a **Pull request** so that your changes can be reviewed

Be sure to fetch and merge from upstream before making a pull request.

## Acknowledgement
Special thanks to [@alexdevonport](https://github.com/alexdevonport) for contributions.

## License
[MIT License](https://github.com/JaredMejia/dadra/blob/main/LICENSE)

## BibTeX
```bibtex
@article{JaredMejia,
  title={DaDRA},
  author={Mejia, Jared},
  journal={GitHub. Note: https://github.com/JaredMejia/dadra},
  volume={1},
  year={2021}
}
```
