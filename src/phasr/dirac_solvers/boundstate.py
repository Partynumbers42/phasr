from .. import constants
from ..config import local_paths
from .base import radial_dirac_eq_norm, initial_values_norm, solver_settings, default_boundstate_settings

from ..utility.math import optimise_radius_highenergy_continuation,derivative
from ..utility.spliner import save_and_load
from ..utility.continuer import highenergy_continuation_exp

from ..nuclei.parameterizations.coulomb import energy_coulomb_nk

import numpy as np
pi = np.pi

from scipy.integrate import solve_ivp, quad
import copy

class boundstates():
    def __init__(self,nucleus,kappa,lepton_mass,
                 bindingenergy_limit_lower=None, bindingenergy_limit_upper=0.,
                 **args):
        
        self.name = nucleus.name
        self.nucleus_type = nucleus.nucleus_type
        self.Z = nucleus.total_charge
        self.kappa = kappa
        self.lepton_mass=lepton_mass
        
        self._current_principal_quantum_number = -self.kappa if self.kappa<0 else self.kappa+1 
        self.principal_quantum_numbers=[]
        self.principal_quantum_numbers.append(self._current_principal_quantum_number)
        
        self.inital_boundstate_settings = copy.copy(default_boundstate_settings)
        for key in args:
            if key in self.inital_boundstate_settings:
                self.inital_boundstate_settings[key]=args[key] #given keywords overwrite defaults
        
        self.update_solver_setting()
        
        self.nucleus = nucleus
        self.Vmin = nucleus.Vmin
        
        self.bindingenergy_limit_lower = bindingenergy_limit_lower
        self.bindingenergy_limit_upper = bindingenergy_limit_upper
        
        if self.bindingenergy_limit_lower is None:
            if self.Vmin*constants.hc!=-np.inf:
                self.bindingenergy_limit_lower=np.max([self.Vmin*constants.hc + self.solver_setting.energy_precision ,-2*self.lepton_mass])
            elif self.nucleus_type=="coulomb":
                self.bindingenergy_limit_lower=-self.lepton_mass+self.solver_setting.energy_precision
            else: 
                raise ValueError('non-coulomb potentials with r->0: V(r)->-inf  not supported')
        
        self._current_bindingenergy_limit_lower=self.bindingenergy_limit_lower
        self.energy_levels=[]
        
        self.find_next_solution()
    
    def update_solver_setting(self):
        energy_norm = np.abs(energy_coulomb_nk(self._current_principal_quantum_number,self.kappa,self.Z,self.lepton_mass)-self.lepton_mass)
        self.solver_setting = solver_settings(energy_norm=energy_norm,**self.inital_boundstate_settings)
        if self.solver_setting.verbose:
            print("r0=",self.solver_setting.beginning_radius,"fm")
            print("rc=",self.solver_setting.critical_radius,"fm")
            print("rinf=",self.solver_setting.asymptotic_radius,"fm")
            print("dr=",self.solver_setting.radius_optimise_step,"fm")
            print("dE=",self.solver_setting.energy_precision,"MeV")
    
    def find_next_solution(self,**args):
        for key in args:
            if hasattr(self.solver_setting,key):
                setattr(self.solver_setting,key,args[key])
        
        self.find_next_energy_level()
        self.solve_IVP_at_current_energy()
    
    def remove_last_energy_level(self):
        if len(self.energy_levels)>0:
            self.energy_levels=self.energy_levels[:-1]
            if len(self.energy_levels)-1==len(self.principal_quantum_numbers):
                self.principal_quantum_numbers=self.principal_quantum_numbers[:-1]
                self._current_principal_quantum_number-=1
                self._current_bindingenergy_limit_lower=self.energy_levels[-1]+self.solver_setting.energy_precision
                self.update_solver_setting()
    
    def find_next_energy_level(self):
        
        if len(self.energy_levels)==len(self.principal_quantum_numbers):
            self._current_principal_quantum_number+=1
            self.principal_quantum_numbers.append(self._current_principal_quantum_number)
            self._current_bindingenergy_limit_lower=self.energy_levels[-1]+self.solver_setting.energy_precision
            self.update_solver_setting()
        
        path=local_paths.energy_path+self.name+"_"+state_name(self._current_principal_quantum_number,self.kappa)+"_m"+str(self.lepton_mass)+".txt" # add more parameters, fct solver_setting to str
        
        self._current_bindingenergy = save_and_load(path,self.solver_setting.renew,self.solver_setting.save,self.solver_setting.verbose,fmt='%.50e',fct=find_bindingenergy,tracked_params=self.solver_setting.as_dict(),nucleus=self.nucleus,bindingenergy_limit_lower=self._current_bindingenergy_limit_lower,bindingenergy_limit_upper=self.bindingenergy_limit_upper,kappa=self.kappa,lepton_mass=self.lepton_mass,solver_setting=self.solver_setting)
        
        self.energy_levels.append(self._current_bindingenergy)
        
        self._current_energy = self._current_bindingenergy + self.lepton_mass
    
    def solve_IVP_at_current_energy(self):
        
        energy_norm=self.solver_setting.energy_norm
        def DGL(r,fct): return radial_dirac_eq_norm(r,fct,potential=self.nucleus.electric_potential,energy=self._current_energy,mass=self.lepton_mass,kappa=self.kappa,energy_norm=energy_norm)  
        
        scale_initial=1 
        
        beginning_radius = self.solver_setting.beginning_radius_norm
        critical_radius = self.solver_setting.critical_radius_norm
        asymptotic_radius = self.solver_setting.asymptotic_radius_norm
        radius_optimise_step = self.solver_setting.radius_optimise_step_norm
                
        initials= scale_initial*initial_values_norm(beginning_radius_norm=beginning_radius,electric_potential_V0=self.Vmin,energy=self._current_energy,mass=self.lepton_mass,kappa=self.kappa,Z=self.Z,energy_norm=energy_norm,nucleus_type=self.nucleus_type)
        
        if self.solver_setting.verbose:
            print('y0=',initials)
        
        radial_dirac = solve_ivp(DGL, (beginning_radius,asymptotic_radius), initials, dense_output=True, method=self.solver_setting.method, atol=self.solver_setting.atol, rtol=self.solver_setting.rtol)

        def wavefct_g_low(x): return radial_dirac.sol(x)[0]
        def wavefct_f_low(x): return radial_dirac.sol(x)[1]
        
        critical_radius = optimise_radius_highenergy_continuation(wavefct_g_low,critical_radius,radius_optimise_step,beginning_radius)
        critical_radius = optimise_radius_highenergy_continuation(wavefct_f_low,critical_radius,radius_optimise_step,beginning_radius)
        
        def wavefct_g_unnormalised(r,rcrit=critical_radius,wavefct_g_low=wavefct_g_low):
            r_arr = np.atleast_1d(r)
            g = np.zeros(len(r_arr))
            mask_r = r_arr<=rcrit
            if np.any(mask_r):
                g[mask_r]=wavefct_g_low(r_arr[mask_r])
            if np.any(~mask_r):
                G_crit=wavefct_g_low(rcrit)
                dG_crit=derivative(wavefct_g_low,1e-6)(rcrit)
                g[~mask_r]=highenergy_continuation_exp(r_arr[~mask_r],rcrit,G_crit,dG_crit,limit=0,t=0)
            if np.isscalar(r):
                g=g[0]
            return g
        
        def wavefct_f_unnormalised(r,rcrit=critical_radius,wavefct_f_low=wavefct_f_low):
            r_arr = np.atleast_1d(r)
            f = np.zeros(len(r_arr))
            mask_r = r_arr<=rcrit
            if np.any(mask_r):
                f[mask_r]=wavefct_f_low(r_arr[mask_r])
            if np.any(~mask_r):
                G_crit=wavefct_f_low(rcrit)
                dG_crit=derivative(wavefct_f_low,1e-6)(rcrit)
                f[~mask_r]=highenergy_continuation_exp(r_arr[~mask_r],rcrit,G_crit,dG_crit,limit=0,t=0)
            if np.isscalar(r):
                f=f[0]
            return f
        
        def integrand_norm(x): return wavefct_g_unnormalised(x)**2 + wavefct_f_unnormalised(x)**2
        
        int_low,_=quad(integrand_norm,beginning_radius,critical_radius,limit=1000) 
        int_high,_=quad(integrand_norm,critical_radius,np.inf,limit=1000) 
        norm_sq = int_low + int_high
        if not (norm_sq < np.inf):
            norm_sq=1
            print("function could not be normalized as norm is not finite")
        
        def wavefct_g(r,wavefct_g_unnormalised=wavefct_g_unnormalised,energy_norm=energy_norm,norm_sq=norm_sq): return wavefct_g_unnormalised(r*energy_norm/constants.hc)/np.sqrt(norm_sq*self.lepton_mass/energy_norm)
        def wavefct_f(r,wavefct_f_unnormalised=wavefct_f_unnormalised,energy_norm=energy_norm,norm_sq=norm_sq): return wavefct_f_unnormalised(r*energy_norm/constants.hc)/np.sqrt(norm_sq*self.lepton_mass/energy_norm)
        
        setattr(self,"wavefunction_g_"+state_name(self._current_principal_quantum_number,self.kappa),wavefct_g)
        setattr(self,"wavefunction_f_"+state_name(self._current_principal_quantum_number,self.kappa),wavefct_f)

def state_name(n,kappa):
    j=np.abs(kappa)-0.5
    l=kappa if kappa>0 else -kappa-1
    l_label = 's' if l==0 else 'p' if l==1 else 'd' if l==2 else 'f' if l==3 else 'g' if l==4 else '_l'+str(l)+'_'
    return str(n)+l_label+str(int(2*j))+'2'

def find_bindingenergy(nucleus,bindingenergy_limit_lower,bindingenergy_limit_upper,kappa,lepton_mass,solver_setting):
    
    energy_limit_lower = bindingenergy_limit_lower + lepton_mass
    energy_limit_upper = bindingenergy_limit_upper + lepton_mass
    
    verbose=solver_setting.verbose
    energy_precision=solver_setting.energy_precision
    
    if verbose:
        print('Searching for boundstate in the range of: [',bindingenergy_limit_lower,',',bindingenergy_limit_upper,']')
    bindingenergy=-np.inf
    
    if bindingenergy_limit_upper<=bindingenergy_limit_lower:
        raise ValueError("lower energy limit needs to be smaller than upper energy limit")
    while (bindingenergy_limit_upper-bindingenergy_limit_lower)>energy_precision:
        energy_limit_lower, energy_limit_upper = find_asymptotic_flip(nucleus,energy_limit_lower,energy_limit_upper,kappa,lepton_mass,solver_setting)
        bindingenergy_limit_lower, bindingenergy_limit_upper  = energy_limit_lower - lepton_mass, energy_limit_upper - lepton_mass
        bindingenergy=(bindingenergy_limit_upper+bindingenergy_limit_lower)/2
        if verbose:
            print('[',bindingenergy_limit_lower,',',bindingenergy_limit_upper,']->',bindingenergy)
    
    return bindingenergy

def find_asymptotic_flip(nucleus,energy_limit_lower,energy_limit_upper,kappa,lepton_mass,solver_setting):

    scale_initial=1e0
    
    beginning_radius=solver_setting.beginning_radius_norm
    asymptotic_radius=solver_setting.asymptotic_radius_norm
    energy_norm=solver_setting.energy_norm
    
    enery_limit_lower_new=energy_limit_lower
    enery_limit_upper_new=energy_limit_upper

    first=True
    for energy in np.linspace(energy_limit_lower,energy_limit_upper,solver_setting.energy_subdivisions):
        
        def DGL(r,y): return radial_dirac_eq_norm(r,y,potential=nucleus.electric_potential,energy=energy,mass=lepton_mass,kappa=kappa,energy_norm=energy_norm,contain=True)
        initials= scale_initial*initial_values_norm(beginning_radius_norm=beginning_radius,electric_potential_V0=nucleus.Vmin,energy=energy,mass=lepton_mass,kappa=kappa,Z=nucleus.Z,energy_norm=energy_norm,nucleus_type=nucleus.nucleus_type,contain=True)
        
        #if solver_setting.verbose:
        #    print('y0=',initials)
        
        radial_dirac = solve_ivp(DGL, (beginning_radius,asymptotic_radius), initials, t_eval=np.array([asymptotic_radius]), method=solver_setting.method, atol=solver_setting.atol, rtol=solver_setting.rtol)
        sign=np.sign(radial_dirac.y[0][-1])
        
        if first:
            sign_ini=sign
            first=False

        if sign == -sign_ini:
            if energy<enery_limit_upper_new:
                enery_limit_upper_new=energy
            return (enery_limit_lower_new, enery_limit_upper_new)
        else:
            if energy>enery_limit_lower_new:
                enery_limit_lower_new=energy
                
    raise  ValueError("No sign flip found between energy_limit_lower and enery_limit_upper, adjust energyrange or increase subdivisions")


