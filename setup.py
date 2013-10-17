# setup.py
# only if building in place: ``python setup.py build_ext --inplace``

from distutils.core import setup, Extension
import numpy


setup(
    name='CommonSE',
    version='1.0.0',
    description='Common utilities for NREL WISDEM',
    author='S. Andrew Ning',
    author_email='andrew.ning@nrel.gov',
    packages=['wisdem', 'wisdem.commonse'],
    package_dir={'': 'src'},
    license='Apache License, Version 2.0',
    ext_modules=[Extension('wisdem.commonse._akima', ['src/wisdem/commonse/akima.c'],
    include_dirs=[numpy.get_include()])]
)