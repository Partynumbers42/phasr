from .. import constants
from ..config import local_paths
from .base import radial_dirac_eq_norm, initial_values_norm, solver_settings, default_continuumstate_settings

import numpy as np
pi = np.pi

from scipy.integrate import solve_ivp, quad
import copy

class continuumstates():
    def __init__(self,nucleus,kappa,energy,lepton_mass,
                 **args):
        
        self.name = nucleus.name
        self.nucleus_type = nucleus.nucleus_type
        self.Z = nucleus.total_charge
        self.kappa = kappa
        self.energy = energy
        self.lepton_mass=lepton_mass
        
        self.inital_continuumstate_settings = copy.copy(default_continuumstate_settings)
        for key in args:
            if key in self.inital_continuumstate_settings:
                self.inital_continuumstate_settings[key]=args[key] #given keywords overwrite defaults
        
        self.update_solver_setting()
        
        self.nucleus = nucleus
        self.Vmin = nucleus.Vmin
        
        pass
    
    
    def update_solver_setting(self):
        energy_norm = 1 # TODO ?
        self.solver_setting = solver_settings(energy_norm=energy_norm,**self.inital_continuumstate_settings)
        if self.solver_setting.verbose:
            print("r0=",self.solver_setting.beginning_radius,"fm")
            print("rc=",self.solver_setting.critical_radius,"fm")
            print("rinf=",self.solver_setting.asymptotic_radius,"fm")
            print("dr=",self.solver_setting.radius_optimise_step,"fm")
            print("dE=",self.solver_setting.energy_precision,"MeV")