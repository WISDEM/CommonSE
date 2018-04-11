from setuptools import setup, find_packages
# from numpy.distutils.core import setup, Extension


setup(
    name='CommonSE',
    version='0.1.3',
    description='Common utilities for NREL WISDEM',
    author='S. Andrew Ning',
    author_email='andrew.ning@nrel.gov',
    #packages= find_packages(),
    packages=['commonse', 'commonse.static'],
    package_data={'':['*.txt']},
    include_package_data = True,
    package_dir={'': 'src'},
    license='Apache License, Version 2.0',
    install_requires=['akima>=1.0'],
    dependency_links=['https://github.com/andrewning/akima/tarball/master#egg=akima-1.0.0']
    # ext_modules=[Extension('commonse._akima', ['src/commonse/akima.f90'], extra_compile_args=['-O2'])]
)
