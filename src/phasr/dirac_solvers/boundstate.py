from .. import constants
from base import radial_dirac_eq, initial_values

from ..utility.math import optimise_radius_highenergy_continuation

import numpy as np
pi = np.pi

from scipy.integrate import solve_ivp

def flipE(nucleus,energy_limit_lower,enery_limit_upper,kappa,mass,subdivisions=100,beginning_radius=1e-12,asymptotic_radius=2.5,atol=1e-6,rtol=1e-3,method='DOP853'):

    V0=nucleus.Vmin
    V=nucleus.electric_potential
    Z=nucleus.Z
    nucleus_type=nucleus.nucleus_type
    m=mass
    
    enery_limit_lower_new=energy_limit_lower
    enery_limit_upper_new=enery_limit_upper

    first=True
    for energy in np.linspace(energy_limit_lower,enery_limit_upper,subdivisions):

        def DGL(r,fct): return radial_dirac_eq(r,fct,potential=V,energy=energy,mass=m,kappa=kappa)
        initials=initial_values(beginning_radius=beginning_radius,electric_potential_V0=V0,energy=energy,mass=m,kappa=kappa,Z=Z,nucleus_type=nucleus_type)
        radial_dirac = solve_ivp(DGL, (beginning_radius,asymptotic_radius), initials, method=method, t_eval=np.array([asymptotic_radius]), atol=atol, rtol=rtol)
        sign=np.sign(radial_dirac.y[0])

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

class groundstate():
    def __init__(self,nucleus,kappa,mass,energy_limit_lower=None,energy_limit_upper=0.,subdivisions=50,hauptquantenzahl=None,scale_initial=1e0,increase_tol_for_high_kappa=True,optimize=True,rmin_max=1e-3,kappa_crit=7,E_prescision=1e-12,rmin_Z=1e-12,rmax_Z=20,rcrit_Z=15,rinf_Z=800,rshow=None,rpres_Z=1e-2,atol=1e-12,rtol=1e-9,units='alpha m',renew=False,verbose=True,method='DOP853',verboseLoad=True,save=True):
        
        self.energy = None # TODO
        
        # includes SOLVING THE IVP
        pass
    
    def wavefunction_g(self,r):
        return 
    
    def wavefunction_g(self,r):
        return 


def groundstate(nucleus,kappa,mass,energy_limit_lower=None,energy_limit_upper=0.,subdivisions=50,hauptquantenzahl=None,scale_initial=1e0,increase_tol_for_high_kappa=True,optimize=True,rmin_max=1e-3,kappa_crit=7,E_prescision=1e-12,rmin_Z=1e-12,rmax_Z=20,rcrit_Z=15,rinf_Z=800,rshow=None,rpres_Z=1e-2,atol=1e-12,rtol=1e-9,units='alpha m',renew=False,verbose=True,method='DOP853',verboseLoad=True,save=True):
    
    #hauptquantenzahl is the starting n, just for labeling purpose -> include automation!
    if hauptquantenzahl is None:
        hauptquantenzahl = -kappa if kappa<0 else kappa+1 
    
    if verbose:
        print('state:',state_name(hauptquantenzahl,kappa))
    
    #,rmax=1.5,rcrit=1.0,rinf=10
    #Emin=None,Emax=0.,kappa=-1,m=1./alpha_el,Z=1,N=20,E_prescision=1e-12,rmin_Z=1e-12,rmax_Z=20,rinf_Z=800,rpres_Z=1e-2,atol=1e-12,rtol=1e-9,method='DOP853'
    #,atol=1e-12,rtol=1e-9,method='DOP853'

    nucleus_type = nucleus.nucleus_type
    V0=nucleus.Vmin
    V=nucleus.electric_potential
    Z=nucleus.Z
    m=mass
    
    rmax=rmax_Z/Z
    rmin=rmin_Z/Z
    rpres=rpres_Z/Z
    rcrit=rcrit_Z/Z
    rinf=rinf_Z/Z

    if optimize:
        rmin=np.min([rmin**(1./np.abs(kappa)),rmin_max])
        if np.abs(kappa)>=np.abs(kappa_crit):
            if increase_tol_for_high_kappa:
                atol/=1.0e1**(2*(np.abs(kappa)-np.abs(kappa_crit)))
                rtol/=1.0e1**(2*(np.abs(kappa)-np.abs(kappa_crit)))
            scale_initial*=1.0e1**(2*(np.abs(kappa)-np.abs(kappa_crit)))
            if rtol<3e-14:
                if verbose:
                    print("minimum rtol reached, it's capped to ",3e-14)
                rtol=np.max([rtol,3e-14]) #maximum possible rtol

    if energy_limit_lower==None:
        if V0!=-np.inf:
            energy_limit_lower=V0-E_prescision
        elif nucleus.atomtype=="coulomb":
            energy_limit_lower=-m
        else: 
            #energy_limit_lower=-2*m
            raise ValueError('non-coulomb potentials with r->0: V(r)->-inf  not supported')
            
    pathE="./lib/splines/Ebin"+nucleus.name+"_"+state_name(hauptquantenzahl,kappa)[:-2]+"_m"+str(m)+".txt"
    if os.path.exists(pathE) and renew==False:
        with open( pathE, "rb" ) as file:
            E_load = np.loadtxt( file , dtype=float)
            energy=E_load[0]
            if verboseLoad:
                print("ground state energy loaded from ",pathE,"as",energy,"+-",E_load[1])
    else:
        if verbose:
            print('searching for groundstate in the range of: [',energy_limit_lower,',',energy_limit_upper,']')
        energy=-np.inf
        if verboseLoad:
            print("ground state energy not found or forced to recalculate.\nThis may take some time.")
        if energy_limit_upper<=energy_limit_lower:
            raise ValueError("Emin needs to be smaller than Emax")
        while (energy_limit_upper-energy_limit_lower)>E_prescision:
            energy_limit_lower, energy_limit_upper = flipE(nucleus,energy_limit_lower,energy_limit_upper,subdivisions,kappa,m,rmin,rmax,rinf,rpres,atol=atol,rtol=rtol,method=method)
            energy=(energy_limit_upper+energy_limit_lower)/2
            if verbose:
                print('[',energy_limit_lower,',',energy_limit_upper,']->',energy)
        if save:
            with open( pathE, "wb" ) as file:
                np.savetxt(file,np.array([energy,E_prescision]),fmt='%.50e')
                if verboseLoad:
                    print("value saved in ", pathE)
    
    # Consider EisEbin situation <---

    def DGL(r,fct): return radial_dirac_eq(r,fct,potential=V,energy=energy,mass=m,kappa=kappa)
    initials=  scale_initial* initial_values(beginning_radius=beginning_radius,electric_potential_V0=V0,energy=energy,mass=m,kappa=kappa,Z=Z,nucleus_type=nucleus_type)
    radial_dirac = solve_ivp(DGL, (beginning_radius,asymptotic_radius), initials, method=method, dense_output=True, atol=atol, rtol=rtol)
    
    wavefct_G = lambda x: radial_dirac.sol(x)[0]
    wavefct_F = lambda x: radial_dirac.sol(x)[1]
    
    rcit = optimise_radius_highenergy_continuation(wavefct_G,rcrit,1e-3,rmin)
    rcit = optimise_radius_highenergy_continuation(wavefct_F,rcrit,1e-3,rmin)

    def wavefct_G_ultimate(r,R=rcrit):
        
        G_crit=wavefct_G_spl(R)
        dG_crit=deriv(wavefct_G_spl,R,1e-6)
        
        if np.size(r)>1:
            G = 0*r
            if np.size(G[np.where(r<=R)])>0:
                G[np.where(r<=R)] = wavefct_G_spl(r[np.where(r<=R)])
            if np.size(G[np.where(r>R)])>0:
                G[np.where(r>R)]=fs.highenergycontinuation(r[np.where(r>R)],R,G_crit,dG_crit,0,t=0)
        else:
            G=wavefct_G_spl(r) if r<=R else fs.highenergycontinuation(r,R,G_crit,dG_crit,0,t=0)

        return G
    
    def wavefct_F_ultimate(r,R=rcrit):
        
        F_crit=wavefct_F_spl(R)
        dF_crit=deriv(wavefct_F_spl,R,1e-6)
        
        if np.size(r)>1:
            F = 0*r
            if np.size(F[np.where(r<=R)])>0:
                F[np.where(r<=R)] = wavefct_F_spl(r[np.where(r<=R)])
            if np.size(F[np.where(r>R)])>0:
                F[np.where(r>R)]=fs.highenergycontinuation(r[np.where(r>R)],R,F_crit,dF_crit,0,t=0)
        else:
            F=wavefct_F_spl(r) if r<=R else fs.highenergycontinuation(r,R,F_crit,dF_crit,0,t=0)

        return F
    

    # def wavefct_G_ultimate(r,R=rcrit,units=units):
        
    #     G_crit=wavefct_G_spl(R)
    #     dG_crit=deriv(wavefct_G_spl,R,1e-6)

    #     G=fs.highenergycontinuation(r,R,G_crit,dG_crit,0,t=0)
    #     if np.any(r<=R):
    #         G = wavefct_G_spl(r)
    #     else:
    #         G = 0*r
    #     if np.size(G)>1:
    #         if np.size(G[np.where(r>R)])>0:
    #             G[np.where(r>R)]=fs.highenergycontinuation(r[np.where(r>R)],R,G_crit,dG_crit,0,t=0)
    #     return G

    # def wavefct_F_ultimate(r,R=rcrit):

    #     F_crit=wavefct_F_spl(R)
    #     dF_crit=deriv(wavefct_F_spl,R,1e-6)

    #     F=fs.highenergycontinuation(r,R,F_crit,dF_crit,0,t=0)
    #     if np.any(r<=R):
    #         F = wavefct_F_spl(r)
    #     else:
    #         F = 0*r
    #     if np.size(F)>1:
    #         if np.size(F[np.where(r>R)])>0:
    #             F[np.where(r>R)]=fs.highenergycontinuation(r[np.where(r>R)],R,F_crit,dF_crit,0,t=0)
    #     return F
    
    norm1,norm1_pres=quad(lambda x: (wavefct_G_ultimate(x)**2 + wavefct_F_ultimate(x)**2),rmin,rcrit,limit=200) 
    norm2,norm2_pres=quad(lambda x: (wavefct_G_ultimate(x)**2 + wavefct_F_ultimate(x)**2),rcrit,np.inf,limit=200) 
    norm = norm1 + norm2
    #print(norm,norm1_pres,norm1_pres)
    
    def G0(r):
        if units=='alpha m':
            G=wavefct_G_ultimate(r*1)/np.sqrt(norm)
            G*=1
        elif units=='m':
            G=wavefct_G_ultimate(r*alpha_el)/np.sqrt(norm)
            G*=np.sqrt(alpha_el)
        else:
            raise ValueError('unit system not known')
        return G

    def F0(r):
        if units=='alpha m':
            F=wavefct_F_ultimate(r*1)/np.sqrt(norm)
            F*=1
        elif units=='m':
            F=wavefct_F_ultimate(r*alpha_el)/np.sqrt(norm)
            F*=np.sqrt(alpha_el)
        else:
            raise ValueError('unit system not known')
        return F

    if rshow is None:
        rshow=rmax

    r=np.arange(rmin,rshow,rpres)

    if units=='alpha m':
        energy*=1
        if verbose:
            plt.plot(r,G0(r))
            plt.plot(r,F0(r))
    elif units=='m':
        energy*=alpha_el
        if verbose:
            plt.plot(r/alpha_el,G0(r/alpha_el))
            plt.plot(r/alpha_el,F0(r/alpha_el))
    else:
        raise ValueError('unit system not known')

    if verbose:
        print('ground state energy:',energy,units)

    return energy, G0, F0








def state_name(n,kappa):
    j=np.abs(kappa)-0.5
    l=kappa if kappa>0 else -kappa-1
    l_label = 's' if l==0 else 'p' if l==1 else 'd' if l==2 else 'f' if l==3 else 'g('+str(l)+')'
    return str(n)+l_label+str(int(2*j))+'/2'

