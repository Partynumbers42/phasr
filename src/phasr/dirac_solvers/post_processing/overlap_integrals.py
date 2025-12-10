import numpy as np

from .. import boundstates
from .. import continuumstates
from ... import masses,constants

from scipy.integrate import quad

def ejection_energy(initial_lepton_mass,final_lepton_mass,binding_energy,nucleus_mass=np.inf):
    #
    if nucleus_mass < np.inf:
        binding_mass = nucleus_mass + initial_lepton_mass + binding_energy
        return (binding_mass**2 - nucleus_mass**2 + final_lepton_mass**2)/(2*binding_mass)
    else:
        return initial_lepton_mass + binding_energy

def select_density(nucleus_response,response):
    
    if response == 'p':
        return nucleus_response.proton_density
    elif response == 'n':
        return nucleus_response.neutron_density
    elif response == 'ch':
        return nucleus_response.charge_density
    elif response == 'w':
        return nucleus_response.weak_density
    elif response == 'rho2M0p':
        return nucleus_response.rho2M0p
    elif response == 'rho2M0n':
        return nucleus_response.rho2M0n
    elif response == 'rho2Phipp0p':
        return nucleus_response.rho2Phipp0p
    elif response == 'rho2Phipp0n':
        return nucleus_response.rho2Phipp0n
    else:
        raise ValueError("Unphysical response chosen")

def get_dimfactor(response):

    if response in ['p' , 'n' , 'ch' , 'w']:
        dimfactor = (constants.hc/masses.mmu)**3
    elif response in ['rho2M0p', 'rho2M0n', 'rho2Phipp0p', 'rho2Phipp0n']:
        dimfactor = (constants.hc/masses.mmu)**5
    else:
        raise ValueError("Dimension of overlap integral is wrong!")
    
    return dimfactor


def calculate_states(nucleus_potential,kappa_e=-1,recoil=True,nonzero_electron_mass=True,args_boundstate={},args_continuumstate={}):

    mass_muon = masses.mmu
    mass_electron = masses.me if nonzero_electron_mass else 0
    mass_nucleus = nucleus_potential.mass if recoil else np.inf
    
    # muon
    boundstate = boundstates(nucleus_potential,kappa=-1,lepton_mass=masses.mmu,**args_boundstate)
    binding_energy = boundstate.energy_levels[0]
    
    energy = ejection_energy(mass_muon,mass_electron,binding_energy,mass_nucleus)
    
    # electron (me=0)
    continuumstate = continuumstates(nucleus_potential,kappa=kappa_e,energy=energy,lepton_mass=mass_electron,**args_continuumstate)
    continuumstate.solve_IVP()

    return boundstate, continuumstate


def overlap_integral_scalar(nucleus_potential,response,nucleus_response=None,kappa_e=-1,recoil=True,nonzero_electron_mass=True,**args):
    # response = 'p' , 'n' , 'ch' , 'w', 'rho2M0p', 'rho2M0n', 'rho2Phipp0p', 'rho2Phipp0n'
    
    boundstate, continuumstate = calculate_states(nucleus_potential,kappa_e=kappa_e,recoil=recoil,nonzero_electron_mass=nonzero_electron_mass,**args)

    if nucleus_response is None:
        nucleus_response = nucleus_potential

    density = select_density(nucleus_response,response)

    dimfactor = get_dimfactor(response)

    if kappa_e==-1:
        def integrand(r):
            return 1/(2*np.sqrt(2)) * density(r) * dimfactor * ( + continuumstate.wavefct_g(r)*boundstate.wavefunction_g_1s12(r) - continuumstate.wavefct_f(r)*boundstate.wavefunction_f_1s12(r) )
    elif kappa_e==+1:
        def integrand(r):
            return 1/(2*np.sqrt(2)) * density(r) * dimfactor * ( - continuumstate.wavefct_f(r)*boundstate.wavefunction_g_1s12(r) - continuumstate.wavefct_g(r)*boundstate.wavefunction_f_1s12(r) )
    else:
        raise ValueError("Unphysical value for kappa_e choosen")
    
    overlap_integral, _ = quad(integrand,0,np.inf,limit=1000) # in mmu^7/2 fm^-1 for LO, mmu^11/2 for NLO
    
    return overlap_integral *(masses.mmu/constants.hc) # in mmu^5/2 for LO, mmu^9/2 for NLO

def overlap_integral_vector(nucleus_potential,response,nucleus_response=None,kappa_e=-1,recoil=True,nonzero_electron_mass=True,**args):
    # response = 'p' , 'n' , 'ch' , 'w', 'rho2M0p', 'rho2M0n', 'rho2Phipp0p', 'rho2Phipp0n'
    
    boundstate, continuumstate = calculate_states(nucleus_potential,kappa_e=kappa_e,recoil=recoil,nonzero_electron_mass=nonzero_electron_mass,**args)

    if nucleus_response is None:
        nucleus_response = nucleus_potential

    density = select_density(nucleus_response,response)

    dimfactor = get_dimfactor(response)

    if kappa_e==-1:
        def integrand(r):
            return 1/(2*np.sqrt(2)) * density(r) * dimfactor * ( + continuumstate.wavefct_g(r)*boundstate.wavefunction_g_1s12(r) + continuumstate.wavefct_f(r)*boundstate.wavefunction_f_1s12(r) )
    elif kappa_e==+1:
        def integrand(r):
            return 1/(2*np.sqrt(2)) * density(r) * dimfactor * ( - continuumstate.wavefct_f(r)*boundstate.wavefunction_g_1s12(r) + continuumstate.wavefct_g(r)*boundstate.wavefunction_f_1s12(r) )
    else:
        raise ValueError("Unphysical value for kappa_e chosen")
    
    overlap_integral, _ = quad(integrand,0,np.inf,limit=1000) # in mmu^7/2 fm^-1 for LO, mmu^11/2 for NLO
    
    return overlap_integral *(masses.mmu/constants.hc) # in mmu^5/2 for LO, mmu^9/2 for NLO
    
def overlap_integral_dipole(nucleus_potential,nucleus_response=None,kappa_e=-1,recoil=True,nonzero_electron_mass=True,**args):
    
    boundstate, continuumstate = calculate_states(nucleus_potential,kappa_e=kappa_e,recoil=recoil,nonzero_electron_mass=nonzero_electron_mass,**args)

    if nucleus_response is None:
        nucleus_response = nucleus_potential

    electric_field = nucleus_response.electric_field
        
    if kappa_e==-1:
        def integrand(r):
            return 4/(np.sqrt(2)) * electric_field(r)*(constants.hc/masses.mmu)**2 * ( - continuumstate.wavefct_g(r)*boundstate.wavefunction_f_1s12(r) - continuumstate.wavefct_f(r)*boundstate.wavefunction_g_1s12(r) )
    elif kappa_e==+1:
        def integrand(r):
            return 4/(np.sqrt(2)) * electric_field(r)*(constants.hc/masses.mmu)**2 * ( + continuumstate.wavefct_f(r)*boundstate.wavefunction_f_1s12(r) - continuumstate.wavefct_g(r)*boundstate.wavefunction_g_1s12(r) )
    else:
        raise ValueError("Unphysical value for kappa_e chosen")
        
    overlap_integral, _ = quad(integrand,0,np.inf,limit=1000) # in mmu^5/2 fm^-1
    
    return overlap_integral *(masses.mmu/constants.hc) # in mmu^3/2