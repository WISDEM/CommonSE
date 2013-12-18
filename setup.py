# setup.py
# only if building in place: ``python setup.py build_ext --inplace``

from setuptools import setup
from numpy.distutils.core import setup, Extension
# import numpy


setup(
    name='CommonSE',
    version='1.0.0',
    description='Common utilities for NREL WISDEM',
    author='S. Andrew Ning',
    author_email='andrew.ning@nrel.gov',
    packages=['commonse'],
    package_dir={'': 'src'},
    license='Apache License, Version 2.0',
    install_requires=['akima>=1.0'],
    dependency_links=['https://github.com/andrewning/akima/tarball/master#egg=akima-1.0.0']
    # ext_modules=[Extension('commonse._akima', ['src/commonse/akima.f90'], extra_compile_args=['-O2'])]
)
