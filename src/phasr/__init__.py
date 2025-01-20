"""Package to calculate scattering phase shifts for arbitrary radial potentials using the phase shift method as well as resulting crosssections as mainly used in the context of elastic electron nucleus scattering."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("phasr")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Frederic Noel"
__email__ = "noel@ipt.unibe.ch"

# define calls from top level
from .physical_constants import constants, masses, trafos
from .nuclei import nucleus
from .dirac_solvers import boundstate_settings, boundstates
