from ... import constants

from ..continuumstate import continuumstates
from ...nuclei.parameterisations.coulomb import delta_coulomb, eta_coulomb

import numpy as np
pi = np.pi

from scipy.special import lpmv as associated_legendre

from ...utility.math import momentum

def recoil_quantities(energy_lab,theta_lab,mass):
    energy_CMS=energy_lab*(1.-energy_lab/mass)
    theta_CMS=theta_lab+(energy_lab/mass)*np.sin(theta_lab)
    scalefactor_crosssection_CMS = 1+(2*energy_lab/mass)*np.cos(theta_lab)
    return energy_CMS, theta_CMS, scalefactor_crosssection_CMS

def crosssection_lepton_nucleus_scattering(energy,theta,nucleus,lepton_mass=0,subtractions=3,recoil=True,N_partial_waves=20,phase_difference_limit=0,**args):
    
    nucleus_mass=nucleus.mass
    charge = nucleus.total_charge

    if recoil:
        energy, theta, scale_crosssection = recoil_quantities(energy,theta,nucleus_mass)
        print('E=',energy,'MeV')
    else:
        scale_crosssection = 1
    
    phase_shifts = {}
    phase_difference_gr0 = True
    for kappa in np.arange(-1,-(N_partial_waves+1+1),-1,dtype=int):
        if phase_difference_gr0:    
            partial_wave_kappa = continuumstates(nucleus,kappa,energy,lepton_mass,**args)
            partial_wave_kappa.extract_phase_shift()
            phase_shifts[kappa] = partial_wave_kappa.phase_shift
            if -kappa < N_partial_waves+1:
                if lepton_mass==0:
                    phase_shifts[-kappa] = phase_shifts[kappa]
                else:
                    partial_wave_mkappa = continuumstates(nucleus,-kappa,energy,lepton_mass,**args)
                    partial_wave_mkappa.extract_phase_shift()
                    phase_shifts[-kappa] = partial_wave_mkappa.phase_shift
                if np.abs(partial_wave_kappa.phase_difference)<=phase_difference_limit:
                    phase_difference_gr0 = False
                    print("phase differences set to zero after kappa=",kappa)
        else:
            eta_regular = eta_coulomb(kappa,charge,energy,lepton_mass,reg=+1)
            phase_shifts[kappa] = delta_coulomb(kappa,charge,energy,lepton_mass,reg=+1,pass_eta=eta_regular) + 0
            if -kappa < N_partial_waves+1:
                if lepton_mass==0:
                    phase_shifts[-kappa] = phase_shifts[kappa]
                else:
                    phase_shifts[-kappa] = delta_coulomb(-kappa,charge,energy,lepton_mass,reg=+1,pass_eta=eta_regular) + 0

    nonspinflip = nonspinflip_amplitude(energy,theta,lepton_mass,N_partial_waves,subtractions,phase_shifts)
    
    if lepton_mass==0:
        crosssection = (1+np.tan(theta/2)**2)*np.abs(nonspinflip)**2
    else:
        print('Warning: m!=0 not fully implmented yet, may have unexpected behaviour')
        spinflip = spinflip_amplitude(energy,theta,lepton_mass,N_partial_waves,subtractions,phase_shifts)
        crosssection = np.abs(nonspinflip)**2 + np.abs(spinflip)**2
        # TODO subtract sumrule used for m=0

    return scale_crosssection * crosssection

def nonspinflip_amplitude(energy,theta,lepton_mass,N_partial_waves,subtractions,phase_shifts):
    k=momentum(energy,lepton_mass)
    amplitude=0
    for kappa in np.arange(0,N_partial_waves-subtractions+1,dtype=int): 
        coefficient=coefficient_nonspinflip_amplitude(kappa,subtractions,N_partial_waves,phase_shifts)
        #print("{:d} {:.5f} {:.5f}".format(kappa,np.real(coefficient),np.imag(coefficient)))
        amplitude+=coefficient*(associated_legendre(0,kappa,np.cos(theta)))
    return (amplitude/((1-np.cos(theta))**subtractions))/(2j*k)

def coefficient_nonspinflip_amplitude(kappa,subtractions,N_partial_waves,phase_shifts):

    if kappa<0:
        raise ValueError("only defined for kappa >= 0")
        
    if subtractions>0:
        last_coefficient_kappa = coefficient_nonspinflip_amplitude(kappa,subtractions-1,N_partial_waves,phase_shifts)
        if N_partial_waves-subtractions>=kappa>0:
            last_coefficient_kappap1 = coefficient_nonspinflip_amplitude(kappa+1,subtractions-1,N_partial_waves,phase_shifts)
            last_coefficient_kappam1 = coefficient_nonspinflip_amplitude(kappa-1,subtractions-1,N_partial_waves,phase_shifts)
            this_coefficient_kappa = last_coefficient_kappa - ((kappa+1)/(2*kappa+3))*last_coefficient_kappap1 - ((kappa)/(2*kappa-1))*last_coefficient_kappam1
        elif kappa==0:
            last_coefficient_kappap1 = coefficient_nonspinflip_amplitude(kappa+1,subtractions-1,N_partial_waves,phase_shifts)
            this_coefficient_kappa = last_coefficient_kappa - ((kappa+1)/(2*kappa+3))*last_coefficient_kappap1
        else:
            raise ValueError("only defined for kappa <= Nmax - m")
    else:
        if N_partial_waves>=kappa>0:
            this_coefficient_kappa = kappa*np.exp(2j*phase_shifts[kappa])+(kappa+1)*np.exp(2j*phase_shifts[-(kappa+1)])
        elif kappa==0:
            this_coefficient_kappa = (kappa+1)*np.exp(2j*phase_shifts[-(kappa+1)])
        else:
            raise ValueError("only defined for kappa <= Nmax")

    return this_coefficient_kappa
#

def spinflip_amplitude(energy,theta,lepton_mass,N_partial_waves,subtractions,phase_shifts):
    k=momentum(energy,lepton_mass)
    amplitude=0
    for kappa in np.arange(0,N_partial_waves-subtractions+1,dtype=int):
        coefficient=coefficient_spinflip_amplitude(kappa,subtractions,N_partial_waves,phase_shifts)
        amplitude+=coefficient*(associated_legendre(1,kappa,np.cos(theta)))
    return (amplitude/((1-np.cos(theta))**subtractions))/(2j*k)

def coefficient_spinflip_amplitude(kappa,subtractions,N_partial_waves,phase_shifts):
    
    if kappa<0:
        raise ValueError("only defined for kappa >= 0")
        
    if subtractions>0:
        last_coefficient_kappa = coefficient_spinflip_amplitude(kappa,subtractions-1,N_partial_waves,phase_shifts)
        if N_partial_waves-subtractions>=kappa>0:
            last_coefficient_kappap1 = coefficient_spinflip_amplitude(kappa+1,subtractions-1,N_partial_waves,phase_shifts)
            last_coefficient_kappam1 = coefficient_spinflip_amplitude(kappa-1,subtractions-1,N_partial_waves,phase_shifts)
            this_coefficient_kappa = last_coefficient_kappa - ((kappa+1+1)/(2*kappa+3))*last_coefficient_kappap1 - ((kappa-1)/(2*kappa-1))*last_coefficient_kappam1
        elif kappa==0:
            last_coefficient_kappap1 = coefficient_spinflip_amplitude(kappa+1,subtractions-1,N_partial_waves,phase_shifts)
            this_coefficient_kappa = last_coefficient_kappa - ((kappa+1+1)/(2*kappa+3))*last_coefficient_kappap1
        else:
            raise ValueError("only defined for kappa <= Nmax - m")
    else:
        if N_partial_waves>=kappa>0:
            this_coefficient_kappa = np.exp(2j*phase_shifts[kappa])+np.exp(2j*phase_shifts[-(kappa+1)])
        elif kappa==0:
            this_coefficient_kappa = np.exp(2j*phase_shifts[-(kappa+1)])
        else:
            raise ValueError("only defined for kappa <= Nmax")

    return this_coefficient_kappa
