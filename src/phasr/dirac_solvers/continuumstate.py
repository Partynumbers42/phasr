from .. import constants
from ..config import local_paths
from .base import radial_dirac_eq_norm, initial_values_norm, solver_settings, default_continuumstate_settings

import numpy as np
pi = np.pi

from scipy.integrate import solve_ivp, quad
import copy

from ..nuclei.parameterisations.coulomb import electric_potential_coulomb, f_coulomb, g_coulomb, hyp1f1_coulomb, eta_coulomb, theta_coulomb

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
        
        #self.solve_IVP()
        
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
    
    def extract_phase_shift(self):
        
        energy_norm=self.solver_setting.energy_norm
        def DGL(r,fct): return radial_dirac_eq_norm(r,fct,potential=self.nucleus.electric_potential,energy=self.energy,mass=self.lepton_mass,kappa=self.kappa,energy_norm=energy_norm)  
        
        scale_initial=1 # TODO also other optimisers
        
        beginning_radius = self.solver_setting.beginning_radius_norm
        critical_radius = self.solver_setting.critical_radius_norm
        #asymptotic_radius = self.solver_setting.asymptotic_radius_norm
        
        initials= scale_initial*initial_values_norm(beginning_radius_norm=beginning_radius,electric_potential_V0=self.Vmin,energy=self.energy,mass=self.lepton_mass,kappa=self.kappa,Z=self.Z,energy_norm=energy_norm,nucleus_type=self.nucleus_type)
        
        radial_dirac = solve_ivp(DGL, (beginning_radius,critical_radius), initials,  t_eval=np.array([critical_radius]), method=self.solver_setting.method, atol=self.solver_setting.atol, rtol=self.solver_setting.rtol)

        wavefct_g_critical_radius = radial_dirac.y[0]
        wavefct_f_critical_radius = radial_dirac.y[1]
        
        fraction = regular_irregular_fraction(wavefct_f_critical_radius,wavefct_g_critical_radius,critical_radius,kappa=self.kappa,Z=self.Z,energy=self.energy,mass=self.lepton_mass)
        

def f_highenergy(r,weight_regular,weight_irregular,kappa,Z,energy,mass,pass_hyp1f1_regular=None,pass_hyp1f1_irregular=None,pass_eta_regular=None,pass_eta_irregular=None,alpha_el=constants.alpha_el):
    f_coulomb_regular=f_coulomb(r,kappa,Z,energy,mass,reg=+1,pass_eta=pass_eta_regular,pass_hyp1f1=pass_hyp1f1_regular,alpha_el=alpha_el)
    f_coulomb_irregular=f_coulomb(r,kappa,Z,energy,mass,reg=-1,pass_eta=pass_eta_irregular,pass_hyp1f1=pass_hyp1f1_irregular,alpha_el=alpha_el)
    return weight_regular*f_coulomb_regular + weight_irregular*f_coulomb_irregular

def g_highenergy(r,weight_regular,weight_irregular,energy,kappa,Z,mass,pass_hyp1f1_regular=None,pass_hyp1f1_irregular=None,pass_eta_regular=None,pass_eta_irregular=None,alpha_el=constants.alpha_el):
    g_coulomb_regular=g_coulomb(r,kappa,Z,energy,mass,reg=+1,pass_eta=pass_eta_regular,pass_hyp1f1=pass_hyp1f1_regular,alpha_el=alpha_el)
    g_coulomb_irregular=g_coulomb(r,kappa,Z,energy,mass,reg=-1,pass_eta=pass_eta_irregular,pass_hyp1f1=pass_hyp1f1_irregular,alpha_el=alpha_el)
    return weight_regular*g_coulomb_regular + weight_irregular*g_coulomb_irregular

def regular_irregular_fraction(wavefct_f_radius,wavefct_g_radius,radius,kappa,Z,energy,mass,pass_hyp1f1_regular=None,pass_hyp1f1_irregular=None,pass_eta_regular=None,pass_eta_irregular=None,alpha_el=constants.alpha_el):
    #
    if pass_hyp1f1_regular is None:
        pass_hyp1f1_regular=hyp1f1_coulomb(radius,kappa,Z,energy,mass,reg=+1,alpha_el=alpha_el)
    if pass_hyp1f1_irregular is None:
        pass_hyp1f1_irregular=hyp1f1_coulomb(radius,kappa,Z,energy,mass,reg=-1,alpha_el=alpha_el)
    if pass_eta_regular is None:
        pass_eta_regular=eta_coulomb(kappa,Z,energy,mass,reg=+1,alpha_el=alpha_el)
    if pass_eta_irregular is None:
        pass_eta_irregular=eta_coulomb(kappa,Z,energy,mass,reg=-1,pass_eta_regular=pass_eta_regular,alpha_el=alpha_el)
    #
    f_coulomb_regular=f_coulomb(radius,kappa,Z,energy,mass,reg=+1,pass_eta=pass_eta_regular,pass_hyp1f1=pass_hyp1f1_regular,alpha_el=alpha_el)
    f_coulomb_irregular=f_coulomb(radius,kappa,Z,energy,mass,reg=-1,pass_eta=pass_eta_irregular,pass_hyp1f1=pass_hyp1f1_irregular,alpha_el=alpha_el)
    g_coulomb_regular=g_coulomb(radius,kappa,Z,energy,mass,reg=+1,pass_eta=pass_eta_regular,pass_hyp1f1=pass_hyp1f1_regular,alpha_el=alpha_el)
    g_coulomb_irregular=g_coulomb(radius,kappa,Z,energy,mass,reg=-1,pass_eta=pass_eta_irregular,pass_hyp1f1=pass_hyp1f1_irregular,alpha_el=alpha_el)
    #
    regular=f_coulomb_irregular - g_coulomb_irregular *(wavefct_f_radius/wavefct_g_radius)
    irregular= f_coulomb_regular - g_coulomb_regular*(wavefct_f_radius/wavefct_g_radius)
    #
    return -regular/irregular if irregular!=0 else  -regular*np.inf

def delta_bar(rc,AoB,E,kappa,Z,m,alpha_el=alpha_el,rc_mod_pi=False):
    theta=theta_coulomb(E,kappa,Z,m,alpha_el=alpha_el)
    if rc_mod_pi:
        tan_delta_bar=np.sin(theta)/(AoB+np.cos(theta))
        delta_b=np.arctan(tan_delta_bar)
    else:
        reg = +1
        kE = k_E(E,m)
        delta_c = delta_coulomb(E,kappa,Z,reg=reg,m=m,alpha_el=alpha_el)
        delta_r = delta_1overr(rc,E,kappa,Z,m=m,alpha_el=alpha_el)
        delta_rc = kE*rc + delta_r + delta_c
        delta_rc = mt.angle_shift(delta_rc,1.) #phase is always invariant under 2*pi shift
        mask_denom=AoB*np.cos(delta_rc)+np.cos(theta+delta_rc)==0
        mask=np.logical_and(np.abs(AoB)<np.inf,np.logical_not(mask_denom))
        if np.size(AoB)>1:
            delta_b=0.*AoB
            if np.size(AoB[mask]>0):
                tan_delta_bar0=(AoB[mask]*np.sin(delta_rc[mask])+np.sin(theta+delta_rc[mask]))/(AoB[mask]*np.cos(delta_rc[mask])+np.cos(theta+delta_rc[mask]))
                delta_b0=np.arctan(tan_delta_bar0)
                delta_b[mask]=delta_b0-delta_rc
            if np.size(AoB[np.logical_not(mask)]>0):
                if np.size(AoB[np.abs(AoB)>=np.inf]>0):
                    delta_b[np.abs(AoB)>=np.inf]=0
                if np.size(AoB[mask_denom]>0):
                    delta_b[mask_denom]=pi/2-delta_rc[mask_denom]
        else:
            if mask:
                tan_delta_bar0=(AoB*np.sin(delta_rc)+np.sin(theta+delta_rc))/(AoB*np.cos(delta_rc)+np.cos(theta+delta_rc))
                delta_b0=np.arctan(tan_delta_bar0)
                delta_b=delta_b0-delta_rc
            else:
                if np.abs(AoB)>=np.inf:
                    delta_b=0
                else:
                    delta_b=pi/2-delta_rc
        delta_b=mt.angle_shift(delta_b,1./2.)
    return delta_b