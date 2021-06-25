from setuptools import find_packages, setup
import setuptools

setup(
    name="dadra",
    packages=setuptools.find_packages(),
    license="MIT",
    version="0.0.2",
    description="Library for Data-Driven Reachability Analysis",
    author="Jared Mejia",
    author_email="jaredmejia@berkeley.edu",
    url="https://github.com/JaredMejia/dadra",
    install_requires=["cvxpy", "matplotlib", "numpy", "scipy"],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
