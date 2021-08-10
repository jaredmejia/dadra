from setuptools import find_packages, setup
import setuptools

from os import path

this_dir = path.abspath(path.dirname(__file__))
with open(path.join(this_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="dadra",
    packages=setuptools.find_packages(),
    license="MIT",
    version="0.1.1",
    description="Library for Data-Driven Reachability Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jared Mejia",
    author_email="jaredmejia@berkeley.edu",
    url="https://github.com/JaredMejia/dadra",
    install_requires=["cvxpy", "matplotlib", "numpy", "scikit-learn", "scipy", "tqdm"],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
