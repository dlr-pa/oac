# Setup file for project OpenAirClim


import os
from setuptools import setup

about = {}
with open("openairclim/__about__.py", mode="r", encoding="utf8") as fp:
    exec(fp.read(), about)


def read(fname):
    """Utility function for reading the README file

    Args:
        fname (str): file name

    Returns:
        str: content of the file
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="openairclim",
    version=about["__version__"],
    author=about["__author__"],
    author_email=about["__email__"],
    description=("This setup.py file installs OpenAirClim."),
    keywords=["climate", "aviation"],
    license=about["__license__"],
    url=about["__url__"],
    packages=["openairclim"],
    long_description=read("README.md"),
    classifiers=[
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Programming Language :: Python :: 3 :: Only",
        "Development Status :: 3 - Alpha",
    ],
    python_requires="==3.11.5",
    install_requires=[
        "setuptools",
        "joblib",
        "numpy",
        "pandas",
        "toml",
        "matplotlib",
        "cf-units",
        "xarray",
        "netcdf4",
        "scipy",
        "deepmerge",
    ],
    extras_require={
        "dev": [
            "platformdirs",
            "black",
            "lazy-object-proxy",
            "jupyterlab",
            "openssl",
            "sphinx",
            "ipykernel",
            "scikit-learn",
            "pytest-cov",
            "prospector",
            "pytest-httpserver",
            "ca-certificates",
            "certifi",
            "pylint",
            "wrapt",
            "mypy",
            "beautifulsoup4",
            "bottleneck",
            "cartopy",
            "pytest",
            "pyroma",
            "isort",
            "ipympl",
            "openpyxl",
        ]
    },
)
