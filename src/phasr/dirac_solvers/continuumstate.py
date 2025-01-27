from .. import constants
from ..config import local_paths
from .base import radial_dirac_eq_norm, initial_values_norm, solver_settings, default_continuumstate_settings

import numpy as np
pi = np.pi

from ..utility.math import momentum, angle_shift_mod_pi, derivative

from scipy.integrate import solve_ivp
import copy

from ..nuclei.parameterisations.coulomb import electric_potential_coulomb, f_coulomb, g_coulomb, hyper1f1_coulomb, eta_coulomb, theta_coulomb, delta_coulomb, delta_1overr

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
        
        self.update_eta_coulomb()

        self.initialize_critical_radius() #potentially time intensive -> make it possible to overwrite
        
        self.update_solver_setting()
        
        self.update_hyper1f1_coulomb_at_critical_radius()
        
    def initialize_critical_radius(self):
        r=np.arange(self.inital_continuumstate_settings['beginning_radius'],self.inital_continuumstate_settings['asymptotic_radius'],self.inital_continuumstate_settings['radius_optimise_step'])
        potential_coulomb_diff=(self.nucleus.electric_potential(r)-electric_potential_coulomb(r,self.Z))/electric_potential_coulomb(r,self.Z)
        r_coulomb = r[np.abs(potential_coulomb_diff)<1e-6]
        for rc in r_coulomb:
            if np.all(np.abs(potential_coulomb_diff[r>=rc])<1e-6):
                self.inital_continuumstate_settings['critical_radius'] = rc
                break
    
    def update_solver_setting(self):
        energy_norm = self.energy # no scaling with Z or kappa ?
        self.solver_setting = solver_settings(energy_norm=energy_norm,**self.inital_continuumstate_settings)
        if self.solver_setting.verbose:
            print("r0=",self.solver_setting.beginning_radius,"fm")
            print("rc=",self.solver_setting.critical_radius,"fm")
            
    def update_eta_coulomb(self):
        self.pass_eta_regular=eta_coulomb(self.kappa,self.Z,self.energy,self.lepton_mass,reg=+1,alpha_el=constants.alpha_el)
        self.pass_eta_irregular=eta_coulomb(self.kappa,self.Z,self.energy,self.lepton_mass,reg=-1,pass_eta_regular=self.pass_eta_regular,alpha_el=constants.alpha_el)

    def update_hyper1f1_coulomb_at_critical_radius(self):
        self.pass_hyper1f1_regular=hyper1f1_coulomb(self.solver_setting.critical_radius,self.kappa,self.Z,self.energy,self.lepton_mass,reg=+1,alpha_el=constants.alpha_el)
        self.pass_hyper1f1_irregular=hyper1f1_coulomb(self.solver_setting.critical_radius,self.kappa,self.Z,self.energy,self.lepton_mass,reg=-1,alpha_el=constants.alpha_el)

    def solve_IVP(self):
        
        energy_norm=self.solver_setting.energy_norm
        def DGL(r,fct): return radial_dirac_eq_norm(r,fct,potential=self.nucleus.electric_potential,energy=self.energy,mass=self.lepton_mass,kappa=self.kappa,energy_norm=energy_norm)  
        
        scale_initial=1 # TODO also other optimisers
        
        beginning_radius = self.solver_setting.beginning_radius_norm
        critical_radius_norm = self.solver_setting.critical_radius_norm
        critical_radius = self.solver_setting.critical_radius
        
        initials= scale_initial*initial_values_norm(beginning_radius_norm=beginning_radius,electric_potential_V0=self.Vmin,energy=self.energy,mass=self.lepton_mass,kappa=self.kappa,Z=self.Z,energy_norm=energy_norm,nucleus_type=self.nucleus_type)
        
        if self.solver_setting.verbose:
            print("y0=",initials)
        
        radial_dirac = solve_ivp(DGL, (beginning_radius,critical_radius_norm), initials, dense_output=True, method=self.solver_setting.method, atol=self.solver_setting.atol, rtol=self.solver_setting.rtol)

        def wavefct_g_low(x): return radial_dirac.sol(x)[0]
        def wavefct_f_low(x): return radial_dirac.sol(x)[1]
        
        wavefct_g_critical_radius = wavefct_g_low(critical_radius_norm)
        wavefct_f_critical_radius = wavefct_f_low(critical_radius_norm)
        
        self.regular_irregular_fraction = regular_irregular_fraction(wavefct_f_critical_radius,wavefct_g_critical_radius,critical_radius,kappa=self.kappa,Z=self.Z,energy=self.energy,mass=self.lepton_mass,pass_hyper1f1_regular=self.pass_hyper1f1_regular,pass_hyper1f1_irregular=self.pass_hyper1f1_irregular,pass_eta_regular=self.pass_eta_regular,pass_eta_irregular=self.pass_eta_irregular)
        
        weight_regular = wavefct_g_critical_radius / g_highenergy(critical_radius,1.,1./self.regular_irregular_fraction,self.kappa,self.Z,self.energy,self.lepton_mass,pass_hyper1f1_regular=self.pass_hyper1f1_regular,pass_hyper1f1_irregular=self.pass_hyper1f1_irregular,pass_eta_regular=self.pass_eta_regular,pass_eta_irregular=self.pass_eta_irregular)
        weight_irregular = weight_regular / self.regular_irregular_fraction
        
        if self.solver_setting.verbose:
            print("A/B=",self.regular_irregular_fraction)
            print(" A =",weight_regular)
            print(" B =",weight_irregular)

        def wavefct_g_unnormalised(r,rcrit=critical_radius_norm,wavefct_g_low=wavefct_g_low,weight_regular=weight_regular,weight_irregular=weight_irregular):
            r_arr = np.atleast_1d(r)
            g = np.zeros(len(r_arr))
            mask_r = r_arr<=rcrit
            if np.any(mask_r):
                g[mask_r]=wavefct_g_low(r_arr[mask_r])
            if np.any(~mask_r):
                g[~mask_r]=g_highenergy(r_arr[~mask_r]*constants.hc/energy_norm,weight_regular,weight_irregular,self.kappa,self.Z,self.energy,self.lepton_mass,pass_eta_regular=self.pass_eta_regular,pass_eta_irregular=self.pass_eta_irregular)
            if np.isscalar(r):
                g=g[0]
            return g
        
        def wavefct_f_unnormalised(r,rcrit=critical_radius_norm,wavefct_f_low=wavefct_f_low,weight_regular=weight_regular,weight_irregular=weight_irregular):
            r_arr = np.atleast_1d(r)
            f = np.zeros(len(r_arr))
            mask_r = r_arr<=rcrit
            if np.any(mask_r):
                f[mask_r]=wavefct_f_low(r_arr[mask_r])
            if np.any(~mask_r):
                f[~mask_r]=f_highenergy(r_arr[~mask_r]*constants.hc/energy_norm,weight_regular,weight_irregular,self.kappa,self.Z,self.energy,self.lepton_mass,pass_eta_regular=self.pass_eta_regular,pass_eta_irregular=self.pass_eta_irregular)
            if np.isscalar(r):
                f=f[0]
            return f
        
        theta=theta_coulomb(self.kappa,self.Z,self.energy,self.lepton_mass,pass_eta_regular=self.pass_eta_regular,pass_eta_irregular=self.pass_eta_irregular)
        norm = np.abs(weight_regular*np.sqrt(1+(1/self.regular_irregular_fraction)**2+2*(1/self.regular_irregular_fraction)*(np.cos(theta)+(self.lepton_mass/self.energy)*np.sin(theta)))) #written in this way to not get to big intermediate values (from A**2 for example)

        if self.solver_setting.verbose:
            print("norm",norm)

        if not (norm < np.inf):
            norm=1
            print("function could not be normalized as norm is not finite")

        def wavefct_g(r,wavefct_g_unnormalised=wavefct_g_unnormalised,energy_norm=energy_norm,norm=norm): return wavefct_g_unnormalised(r*energy_norm/constants.hc)/norm
        def wavefct_f(r,wavefct_f_unnormalised=wavefct_f_unnormalised,energy_norm=energy_norm,norm=norm): return wavefct_f_unnormalised(r*energy_norm/constants.hc)/norm

        self.wavefct_g = wavefct_g
        self.wavefct_f = wavefct_f
    
    def extract_phase_shift(self):
        
        energy_norm=self.solver_setting.energy_norm
        def DGL(r,fct): return radial_dirac_eq_norm(r,fct,potential=self.nucleus.electric_potential,energy=self.energy,mass=self.lepton_mass,kappa=self.kappa,energy_norm=energy_norm)  
        
        scale_initial=1 # TODO also other optimisers
        
        beginning_radius = self.solver_setting.beginning_radius_norm
        critical_radius_norm = self.solver_setting.critical_radius_norm
        critical_radius = self.solver_setting.critical_radius
        
        initials= scale_initial*initial_values_norm(beginning_radius_norm=beginning_radius,electric_potential_V0=self.Vmin,energy=self.energy,mass=self.lepton_mass,kappa=self.kappa,Z=self.Z,energy_norm=energy_norm,nucleus_type=self.nucleus_type)
        
        radial_dirac = solve_ivp(DGL, (beginning_radius,critical_radius_norm), initials,  t_eval=np.array([critical_radius_norm]), method=self.solver_setting.method, atol=self.solver_setting.atol, rtol=self.solver_setting.rtol)

        wavefct_g_critical_radius = radial_dirac.y[0][0]
        wavefct_f_critical_radius = radial_dirac.y[1][0]
        
        self.regular_irregular_fraction = regular_irregular_fraction(wavefct_f_critical_radius,wavefct_g_critical_radius,critical_radius,kappa=self.kappa,Z=self.Z,energy=self.energy,mass=self.lepton_mass,pass_hyper1f1_regular=self.pass_hyper1f1_regular,pass_hyper1f1_irregular=self.pass_hyper1f1_irregular,pass_eta_regular=self.pass_eta_regular,pass_eta_irregular=self.pass_eta_irregular)
        
        if self.solver_setting.verbose:
            print("A/B=",self.regular_irregular_fraction)
        
        self.phase_difference = phase_difference(critical_radius,self.regular_irregular_fraction,kappa=self.kappa,Z=self.Z,energy=self.energy,mass=self.lepton_mass,pass_eta_regular=self.pass_eta_regular,pass_eta_irregular=self.pass_eta_irregular)
        self.phase_shift = delta_coulomb(self.kappa,self.Z,self.energy,self.lepton_mass,reg=+1,pass_eta=self.pass_eta_regular) + self.phase_difference

def g_highenergy(r,weight_regular,weight_irregular,kappa,Z,energy,mass,pass_hyper1f1_regular=None,pass_hyper1f1_irregular=None,pass_eta_regular=None,pass_eta_irregular=None,alpha_el=constants.alpha_el):
    g_coulomb_regular=g_coulomb(r,kappa,Z,energy,mass,reg=+1,pass_eta=pass_eta_regular,pass_hyper1f1=pass_hyper1f1_regular,alpha_el=alpha_el)
    g_coulomb_irregular=g_coulomb(r,kappa,Z,energy,mass,reg=-1,pass_eta=pass_eta_irregular,pass_hyper1f1=pass_hyper1f1_irregular,alpha_el=alpha_el)
    return weight_regular*g_coulomb_regular + weight_irregular*g_coulomb_irregular

def f_highenergy(r,weight_regular,weight_irregular,kappa,Z,energy,mass,pass_hyper1f1_regular=None,pass_hyper1f1_irregular=None,pass_eta_regular=None,pass_eta_irregular=None,alpha_el=constants.alpha_el):
    f_coulomb_regular=f_coulomb(r,kappa,Z,energy,mass,reg=+1,pass_eta=pass_eta_regular,pass_hyper1f1=pass_hyper1f1_regular,alpha_el=alpha_el)
    f_coulomb_irregular=f_coulomb(r,kappa,Z,energy,mass,reg=-1,pass_eta=pass_eta_irregular,pass_hyper1f1=pass_hyper1f1_irregular,alpha_el=alpha_el)
    return weight_regular*f_coulomb_regular + weight_irregular*f_coulomb_irregular

def regular_irregular_fraction(wavefct_f_radius,wavefct_g_radius,radius,kappa,Z,energy,mass,pass_hyper1f1_regular=None,pass_hyper1f1_irregular=None,pass_eta_regular=None,pass_eta_irregular=None,alpha_el=constants.alpha_el):
    #radius in fm 
    #
    if pass_hyper1f1_regular is None:
        pass_hyper1f1_regular=hyper1f1_coulomb(radius,kappa,Z,energy,mass,reg=+1,alpha_el=alpha_el)
    if pass_hyper1f1_irregular is None:
        pass_hyper1f1_irregular=hyper1f1_coulomb(radius,kappa,Z,energy,mass,reg=-1,alpha_el=alpha_el)
    if pass_eta_regular is None:
        pass_eta_regular=eta_coulomb(kappa,Z,energy,mass,reg=+1,alpha_el=alpha_el)
    if pass_eta_irregular is None:
        pass_eta_irregular=eta_coulomb(kappa,Z,energy,mass,reg=-1,pass_eta_regular=pass_eta_regular,alpha_el=alpha_el)
    #
    f_coulomb_regular=f_coulomb(radius,kappa,Z,energy,mass,reg=+1,pass_eta=pass_eta_regular,pass_hyper1f1=pass_hyper1f1_regular,alpha_el=alpha_el)
    f_coulomb_irregular=f_coulomb(radius,kappa,Z,energy,mass,reg=-1,pass_eta=pass_eta_irregular,pass_hyper1f1=pass_hyper1f1_irregular,alpha_el=alpha_el)
    g_coulomb_regular=g_coulomb(radius,kappa,Z,energy,mass,reg=+1,pass_eta=pass_eta_regular,pass_hyper1f1=pass_hyper1f1_regular,alpha_el=alpha_el)
    g_coulomb_irregular=g_coulomb(radius,kappa,Z,energy,mass,reg=-1,pass_eta=pass_eta_irregular,pass_hyper1f1=pass_hyper1f1_irregular,alpha_el=alpha_el)
    #
    regular=f_coulomb_irregular - g_coulomb_irregular *(wavefct_f_radius/wavefct_g_radius)
    irregular= f_coulomb_regular - g_coulomb_regular*(wavefct_f_radius/wavefct_g_radius)
    #
    return -regular/irregular if irregular!=0 else  -regular*np.inf

def phase_difference(radius,fraction,kappa,Z,energy,mass,pass_eta_regular=None,pass_eta_irregular=None,alpha_el=constants.alpha_el):
    #radius in fm 
    theta=theta_coulomb(kappa,Z,energy,mass,pass_eta_regular=pass_eta_regular,pass_eta_irregular=pass_eta_irregular,alpha_el=alpha_el)
    k = momentum(energy,mass)
    specific_coulomb_phase = delta_coulomb(kappa,Z,energy,mass,reg=+1,pass_eta=pass_eta_regular,alpha_el=alpha_el)
    asymptotic_coulomb_phase = delta_1overr(radius,kappa,Z,energy,mass,alpha_el=alpha_el)
    total_coulomb_phase = k*radius/constants.hc + asymptotic_coulomb_phase + specific_coulomb_phase
    total_coulomb_phase = angle_shift_mod_pi(total_coulomb_phase,1.)
    mask_denom=fraction*np.cos(total_coulomb_phase)+np.cos(theta+total_coulomb_phase)==0
    mask=np.logical_and(np.abs(fraction)<np.inf,np.logical_not(mask_denom))
    if mask:
        tan_total_phase=(fraction*np.sin(total_coulomb_phase)+np.sin(theta+total_coulomb_phase))/(fraction*np.cos(total_coulomb_phase)+np.cos(theta+total_coulomb_phase))
        total_phase=np.arctan(tan_total_phase)
        phase_difference=total_phase-total_coulomb_phase
    else:
        if np.abs(fraction)>=np.inf:
            phase_difference=0
        else:
            phase_difference=pi/2-total_coulomb_phase
    return angle_shift_mod_pi(phase_difference,1./2.)
    