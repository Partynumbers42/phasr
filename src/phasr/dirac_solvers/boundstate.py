from .. import constants
from ..config import local_paths
from .base import radial_dirac_eq, initial_values, boundstate_settings

from ..utility.math import optimise_radius_highenergy_continuation,derivative
from ..utility.spliner import saveandload
from ..utility.continuer import highenergy_continuation_exp

import numpy as np
pi = np.pi

from scipy.integrate import solve_ivp, quad

def flipE(nucleus,energy_limit_lower,energy_limit_upper,kappa,lepton_mass,solver_settings=boundstate_settings):

    beginning_radius=solver_settings.beginning_radius
    asymptotic_radius=solver_settings.asymptotic_radius
    
    # get rid of 
    r=np.arange(beginning_radius,solver_settings.critical_radius,solver_settings.radius_precision)
    r=np.append(r,asymptotic_radius)
    # some wrong units still somewhere???
    
    V0=nucleus.Vmin
    V=nucleus.electric_potential
    Z=nucleus.Z
    nucleus_type=nucleus.nucleus_type
    m=lepton_mass
    
    enery_limit_lower_new=energy_limit_lower
    enery_limit_upper_new=energy_limit_upper

    first=True
    for energy in np.linspace(energy_limit_lower,energy_limit_upper,solver_settings.energy_subdivisions):

        def DGL(r,fct): return radial_dirac_eq(r,fct,potential=V,energy=energy,mass=m,kappa=kappa)
        initials=initial_values(beginning_radius=beginning_radius,electric_potential_V0=V0,energy=energy,mass=m,kappa=kappa,Z=Z,nucleus_type=nucleus_type)
        radial_dirac = solve_ivp(DGL, (beginning_radius,asymptotic_radius), initials, t_eval=r , method = solver_settings.method, atol=solver_settings.atol, rtol=solver_settings.rtol)
        #, t_eval=np.linspace(beginning_radius,asymptotic_radius,1000)
        print(radial_dirac)
        print(beginning_radius,asymptotic_radius)
        print(radial_dirac.t)
        print(radial_dirac.y[0])
        print(radial_dirac.y[1])
        
        raise ValueError('break')
        
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

def findE(nucleus,energy_limit_lower,energy_limit_upper,kappa,lepton_mass,solver_settings=boundstate_settings):
    
    verbose=solver_settings.verbose
    energy_precision=solver_settings.energy_precision
    
    if verbose:
        print('Searching for boundstate in the range of: [',energy_limit_lower,',',energy_limit_upper,']')
    energy=-np.inf
    
    if energy_limit_upper<=energy_limit_lower:
        raise ValueError("lower energy limit needs to be smaller than upper energy limit")
    while (energy_limit_upper-energy_limit_lower)>energy_precision:
        energy_limit_lower, energy_limit_upper = flipE(nucleus,energy_limit_lower,energy_limit_upper,kappa,lepton_mass,solver_settings)
        energy=(energy_limit_upper+energy_limit_lower)/2
        if verbose:
            print('[',energy_limit_lower,',',energy_limit_upper,']->',energy)
    
    return energy

class boundstates():
    def __init__(self,nucleus,kappa,lepton_mass,
                 bindingenergy_limit_lower=None, bindingenergy_limit_upper=0.,
                 solver_settings=boundstate_settings,
                 #scale_initial=1e0,increase_tol_for_high_kappa=True,optimize=True,rmin_max=1e-3,kappa_crit=7,
                 #rmin_Z=1e-12,rmax_Z=20,rcrit_Z=15,rinf_Z=800,rpres_Z=1e-2
                 ):
        
        self.kappa = kappa
        self.lepton_mass=lepton_mass
        
        self.name = nucleus.name
        self.nucleus_type = nucleus.nucleus_type
        self.Z = nucleus.Z
        self.Vmin = nucleus.Vmin
        
        self.nucleus = nucleus
        self.solver_settings = solver_settings
        
        self.bindingenergy_limit_lower = bindingenergy_limit_lower
        self.bindingenergy_limit_upper = bindingenergy_limit_upper
        
        if self.bindingenergy_limit_lower is None:
            if self.Vmin*constants.hc!=-np.inf:
                self.bindingenergy_limit_lower=self.Vmin*constants.hc-self.solver_settings.energy_precision
            elif self.nucleus_type=="coulomb":
                self.bindingenergy_limit_lower=-self.lepton_mass-self.solver_settings.energy_precision
            else: 
                raise ValueError('non-coulomb potentials with r->0: V(r)->-inf  not supported')
        
        self.principal_quantum_numbers=[]
        self.energy_levels=[]
        
        self.find_next_energy_level()
        self.solve_IVP()
        
    def find_next_energy_level(self):
        
        if not len(self.principal_quantum_numbers):
            self._current_principal_quantum_number = -self.kappa if self.kappa<0 else self.kappa+1 
            self.principal_quantum_numbers.append(self._current_principal_quantum_number)
            self._current_bindingenergy_limit_lower=self.bindingenergy_limit_lower
        else:
            self._current_principal_quantum_number+=1
            self.principal_quantum_numbers.append(self._current_principal_quantum_number)
            self._current_bindingenergy_limit_lower=self.energy_levels[-1]+self.solver_settings.energy_precision
            
        energy_limit_lower = self._current_bindingenergy_limit_lower + self.lepton_mass
        energy_limit_upper = self.bindingenergy_limit_upper + self.lepton_mass
        
        path=local_paths.energy_path+self.name+"_"+state_name(self._current_principal_quantum_number,self.kappa)[:-2]+"_m"+str(self.lepton_mass)+".txt" # add more parameters, fct solver_settings to str
        
        self._current_energy = saveandload(path,self.solver_settings.renew,self.solver_settings.save,self.solver_settings.verbose,fmt='%.50e',fct=findE,nucleus=self.nucleus,energy_limit_lower=energy_limit_lower,energy_limit_upper=energy_limit_upper,kappa=self.kappa,lepton_mass=self.lepton_mass,solver_settings=self.solver_settings)
        
        self.energy_levels.append(self._current_energy)
        
    
    def solve_IVP(self):
        def DGL(r,fct): return radial_dirac_eq(r,fct,potential=self.nucleus.electric_potential,energy=self._current_energy,mass=self.lepton_mass,kappa=self.kappa)  
        
        scale_initial=1 # TODO also other optimisers
        
        beginning_radius = self.solver_settings.beginning_radius
        critcal_radius = self.solver_settings.critical_radius
        asymptotic_radius = self.solver_settings.asymptotic_radius
                
        initials=  scale_initial*initial_values(beginning_radius=beginning_radius,electric_potential_V0=self.Vmin,energy=self._current_energy,mass=self.lepton_mass,kappa=self.kappa,Z=self.Z,nucleus_type=self.nucleus_type)
        radial_dirac = solve_ivp(DGL, (beginning_radius,asymptotic_radius), initials, dense_output=True, method=self.solver_settings.method, atol=self.solver_settings.atol, rtol=self.solver_settings.rtol)

        def wavefct_g_low(x): return radial_dirac.sol(x)[0]
        def wavefct_f_low(x): return radial_dirac.sol(x)[1]
        
        critical_radius = optimise_radius_highenergy_continuation(wavefct_g_low,critcal_radius,1e-3,beginning_radius)
        critical_radius = optimise_radius_highenergy_continuation(wavefct_f_low,critcal_radius,1e-3,beginning_radius)
        
        def wavefct_g_unnormalised(r,rcrit=critcal_radius,wavefct_g_low=wavefct_g_low):
            r_arr = np.atleast_1d(r)
            g = np.zeros(len(r_arr))
            mask_r = r_arr<=rcrit
            if np.any(mask_r):
                g[mask_r]=wavefct_g_low(r_arr[mask_r])
            if np.any(~mask_r):
                G_crit=wavefct_g_low(rcrit)
                dG_crit=derivative(wavefct_g_low,rcrit,1e-6)
                g[~mask_r]=highenergy_continuation_exp(r_arr[~mask_r],rcrit,G_crit,dG_crit,limit=0,t=0)
            if np.isscalar(r):
                g=g[0]
            return g
        
        def wavefct_f_unnormalised(r,rcrit=critcal_radius,wavefct_f_low=wavefct_f_low):
            r_arr = np.atleast_1d(r)
            f = np.zeros(len(r_arr))
            mask_r = r_arr<=rcrit
            if np.any(mask_r):
                f[mask_r]=wavefct_f_low(r_arr[mask_r])
            if np.any(~mask_r):
                G_crit=wavefct_f_low(rcrit)
                dG_crit=derivative(wavefct_f_low,rcrit,1e-6)
                f[~mask_r]=highenergy_continuation_exp(r_arr[~mask_r],rcrit,G_crit,dG_crit,limit=0,t=0)
            if np.isscalar(r):
                f=f[0]
            return f
        
        def integrand_norm(x): return wavefct_g_unnormalised(x)**2 + wavefct_f_unnormalised(x)**2
        
        int_low,_=quad(integrand_norm,beginning_radius,critcal_radius,limit=1000) 
        int_high,_=quad(integrand_norm,critcal_radius,np.inf,limit=1000) 
        norm = int_low + int_high
        
        def wavefct_g(r,wavefct_g_unnormalised=wavefct_g_unnormalised,norm=norm): return wavefct_g_unnormalised(r)/np.sqrt(norm)
        def wavefct_f(r,wavefct_f_unnormalised=wavefct_f_unnormalised,norm=norm): return wavefct_f_unnormalised(r)/np.sqrt(norm)
        
        setattr(self,"wavefunction_g_"+state_name(self._current_principal_quantum_number,self.kappa)[:-2]+'2',wavefct_g)
        setattr(self,"wavefunction_f_"+state_name(self._current_principal_quantum_number,self.kappa)[:-2]+'2',wavefct_f)

def state_name(n,kappa):
    j=np.abs(kappa)-0.5
    l=kappa if kappa>0 else -kappa-1
    l_label = 's' if l==0 else 'p' if l==1 else 'd' if l==2 else 'f' if l==3 else 'g('+str(l)+')'
    return str(n)+l_label+str(int(2*j))+'/2'

