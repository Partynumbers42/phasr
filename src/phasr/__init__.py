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
from .dirac_solvers import boundstates, continuumstates
from .dirac_solvers import default_boundstate_settings, default_continuumstate_settings
from .dirac_solvers import crosssection_lepton_nucleus_scattering, left_right_asymmetry_lepton_nucleus_scattering
from .dirac_solvers import overlap_integral_scalar, overlap_integral_vector, overlap_integral_dipole
from .dirac_solvers.post_processing.crosssection import optimise_crosssection_precision
from .dirac_solvers.post_processing.left_right_asymmetry import optimise_left_right_asymmetry_precision
from .cross_section_fitter import import_dataset, load_best_fit
from .cross_section_fitter import fit_runner, parallel_fitting_automatic, parallel_fitting_manual