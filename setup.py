from setuptools import find_packages, setup

setup(
    name="dadra",
    packages=find_packages(include=["dadra"]),
    version="0.0.1",
    description="Library for Data-Driven Reachability Analysis",
    author="Jared Mejia",
    author_email="jaredmejia@berkeley.edu",
    url="https://github.com/JaredMejia/dadra",
    license="MIT",
    install_requires=["cvxpy", "matplotlib", "numpy", "scipy"],
)
