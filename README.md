CommonSE is a collection of utilities for common use across the WISDEM set of tools.

Author: [S. Andrew Ning, R. Damiani, and K. Dykes](mailto:nrel.wisdem+commonse@gmail.com)

## Version

This software is a beta version 0.1.0.

## Detailed Documentation

For detailed documentation see <http://wisdem.github.io/CommonSE/>

## Prerequisites

General: NumPy, SciPy, Swig, pyWin32, MatlPlotLib, Lxml, OpenMDAO

## Dependencies

Wind Plant Framework: [FUSED-Wind](http://fusedwind.org) (Framework for Unified Systems Engineering and Design of Wind Plants)

Supporting python packages: Pandas, Algopy, Zope.interface, Sphinx, Xlrd, PyOpt, py2exe, Pyzmq, Sphinxcontrib-bibtex, Sphinxcontrib-zopeext, Numpydoc, Ipython

## Installation

First, clone the [repository](https://github.com/WISDEM/CommonSE)
or download the releases and uncompress/unpack (CommonSE.py-|release|.tar.gz or CommonSE.py-|release|.zip) from the website link at the bottom the [CommonSE site](http://nwtc.nrel.gov/CommonSE).

Install CommonSE with the following command from within an activated OpenMDAO environment.

    $ plugin install

## Run Unit Tests

To check if installation was successful you can try to import the module from within an activated OpenMDAO environment.

    $ python
    > import commonse.environment 

or run the unit tests 

    $ python src/commonse/test/test_environment_gradients.py


For software issues please use <https://github.com/WISDEM/CommonSE/issues>.  For functionality and theory related questions and comments please use the NWTC forum for [Systems Engineering Software Questions](https://wind.nrel.gov/forum/wind/viewtopic.php?f=34&t=1002).

