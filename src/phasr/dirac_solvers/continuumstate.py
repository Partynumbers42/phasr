from .. import constants
from ..config import local_paths
from .base import radial_dirac_eq_norm, initial_values_norm, solver_settings, default_continuumstate_settings

import numpy as np
pi = np.pi

from scipy.integrate import solve_ivp, quad
import copy

from ..nuclei.parameterisations.coulomb import electric_potential_coulomb

class continuumstates():
    def __init__(self,nucleus,kappa,energy,lepton_mass=0,
                 **args):
        
        self.name = nucleus.name
        self.nucleus_type = nucleus.nucleus_type
        self.Z = nucleus.total_charge
        self.kappa = kappa
        self.energy = energy
        self.lepton_mass=lepton_mass
        
        self.nucleus = nucleus
        self.Vmin = nucleus.Vmin
        
        self.inital_continuumstate_settings = copy.copy(default_continuumstate_settings)
        for key in args:
            if key in self.inital_continuumstate_settings:
                self.inital_continuumstate_settings[key]=args[key] #given keywords overwrite defaults
        
        self.initialize_critical_radius() #potentially time intensive
        
        self.update_solver_setting()
        
        self.solve_IVP()
        
    def initialize_critical_radius(self):
        r=np.arange(self.inital_continuumstate_settings['beginning_radius'],self.inital_continuumstate_settings['asymptotic_radius'],self.inital_continuumstate_settings['radius_optimise_step'])
        potential_coulomb_diff=(self.nucleus.electric_potential(r)-electric_potential_coulomb(r,self.Z))/electric_potential_coulomb(r,self.Z)
        r_coulomb = r[np.abs(potential_coulomb_diff)<1e-6]
        for rc in r_coulomb:
            if np.all(np.abs(potential_coulomb_diff[r>=rc])<1e-6):
                self.inital_continuumstate_settings['critical_radius'] = rc
                break
        #if self.inital_continuumstate_settings['verbose']:
        #    print("rc=",self.inital_continuumstate_settings['critical_radius'],'fm')
    
    def update_solver_setting(self):
        energy_norm = self.energy # no scaling with Z or kappa ?
        self.solver_setting = solver_settings(energy_norm=energy_norm,**self.inital_continuumstate_settings)
        if self.solver_setting.verbose:
            print("r0=",self.solver_setting.beginning_radius,"fm")
            print("rc=",self.solver_setting.critical_radius,"fm")
            print("rinf=",self.solver_setting.asymptotic_radius,"fm")
            print("dr=",self.solver_setting.radius_optimise_step,"fm")
            print("dE=",self.solver_setting.energy_precision,"MeV")
    
    def solve_IVP(self):
        
        energy_norm=self.solver_setting.energy_norm
        def DGL(r,fct): return radial_dirac_eq_norm(r,fct,potential=self.nucleus.electric_potential,energy=self.energy,mass=self.lepton_mass,kappa=self.kappa,energy_norm=energy_norm)  
        
        scale_initial=1 # TODO also other optimisers
        
        beginning_radius = self.solver_setting.beginning_radius_norm
        critical_radius = self.solver_setting.critical_radius_norm
        #asymptotic_radius = self.solver_setting.asymptotic_radius_norm
        
        initials= scale_initial*initial_values_norm(beginning_radius_norm=beginning_radius,electric_potential_V0=self.Vmin,energy=self.energy,mass=self.lepton_mass,kappa=self.kappa,Z=self.Z,energy_norm=energy_norm,nucleus_type=self.nucleus_type)
        
        if self.solver_setting.verbose:
            print("y0=",initials)
        
        radial_dirac = solve_ivp(DGL, (beginning_radius,critical_radius), initials, dense_output=True, method=self.solver_setting.method, atol=self.solver_setting.atol, rtol=self.solver_setting.rtol)

        def wavefct_g_low(x): return radial_dirac.sol(x)[0]
        def wavefct_f_low(x): return radial_dirac.sol(x)[1]
        
        # TODO -> high energy continuation 
        
        self.g = wavefct_g_low
        self.f = wavefct_f_low
    