from ... import constants

from ..continuumstate import continuumstates
from ...nuclei.parameterizations.coulomb import delta_coulomb, eta_coulomb

import numpy as np
pi = np.pi

from scipy.special import lpmv as associated_legendre

from ...utility.math import momentum

import itertools

import time, copy

parameter_steps={
    'method' : np.array(['DOP853']), #,'LSODA' 
    'N_partial_waves' : np.append(np.arange(100,50-10,-10),np.arange(50,20-5,-5)),
    'atol' : 10**np.arange(-13,-6+1,1,dtype=float),
    'rtol' : 10**np.arange(-13,-6+1,1,dtype=float),
    'energy_norm': constants.hc*10**np.arange(0,-6-1,-1,dtype=float),
    'phase_difference_limit' : np.append([0],10**np.arange(-15,-6+1,1,dtype=float)),
}

def optimise_crosssection_precision(energy,theta,nucleus,lepton_mass=0,recoil=True,subtractions=3,crosssection_precision=1e-3,jump_forward_dist=1,verbose=False):

    insufficient_args=[]
    
    first=True
    for method,N_partial_waves,atol,rtol,energy_norm,phase_difference_limit in itertools.product(*parameter_steps.values()): 
        
        skip=False
        
        args={'method':method,'N_partial_waves':N_partial_waves,'atol':atol,'rtol':rtol,'energy_norm':energy_norm,'phase_difference_limit':phase_difference_limit} 
        
        for key in args:
            index_key = np.where(parameter_steps[key]==args[key])[0][0]
            if index_key>0:
                args_check = copy.copy(args)
                args_check[key] = parameter_steps[key][index_key-1]
                # skip if more precise calculation was already unsuccesfull
                if args_check in insufficient_args:
                    skip=True
                    insufficient_args.append(args)
                    #print('skiped')
                    break
            
            if index_key<len(parameter_steps[key]) and not first:
                index_best_key = np.where(parameter_steps[key]==best_args[key])[0][0]
                # skip until close to the current best again
                if index_key < index_best_key-jump_forward_dist:
                    skip=True
                    #print('jumped forward')
                    break

        if not skip:

            start_time=time.time()
            crosssection = crosssection_lepton_nucleus_scattering(energy,theta,nucleus,lepton_mass=lepton_mass,recoil=recoil,subtractions=subtractions,verbose=verbose,**args)
            end_time=time.time()
            runtime=end_time-start_time
            
            if first:
                crosssection0=crosssection
                best_time=runtime
                best_args=copy.copy(args)
                first=False
            
            crossections_difference=(crosssection-crosssection0)/crosssection0
        
            if np.all(crossections_difference<crosssection_precision) and (runtime-best_time)/best_time<1e-2:
                if True:#verbose:
                    print('new best:',args)
                    print('time:',runtime,'diff:',np.max(crossections_difference))
                best_time=runtime
                best_args=copy.copy(args)
            
            if np.any(crossections_difference>crosssection_precision):
                insufficient_args.append(args)

    if verbose:
        print('best time:',best_time)    
    
    return best_args

def recoil_quantities(energy_lab,theta_lab,mass):
    energy_CMS=energy_lab*(1.-energy_lab/mass)
    theta_CMS=theta_lab+(energy_lab/mass)*np.sin(theta_lab)
    scalefactor_crosssection_CMS = 1+(2*energy_lab/mass)*np.cos(theta_lab)
    return energy_CMS, theta_CMS, scalefactor_crosssection_CMS

def crosssection_lepton_nucleus_scattering(energy,theta,nucleus,lepton_mass=0,recoil=True,subtractions=3,N_partial_waves=100,verbose=False,phase_difference_limit=0,**args):
    
    args['verbose']=verbose

    nucleus_mass=nucleus.mass
    charge = nucleus.total_charge

    if recoil:
        energy, theta, scale_crosssection = recoil_quantities(energy,theta,nucleus_mass)
        if verbose:
            print('E=',energy,'MeV')
    else:
        scale_crosssection = 1
    
    phase_shifts = {}
    phase_difference_gr0 = True
    # calculate beginning and critical radius only once, since independent on kappa
    if (not ('beginning_radius' in args)) or (not ('critical_radius' in args)):
        initialiser = continuumstates(nucleus,-1,energy,lepton_mass,**args)
    if not 'beginning_radius' in args:
        args['beginning_radius']=initialiser.solver_setting.beginning_radius
    if not 'critical_radius' in args:
        args['critical_radius']=initialiser.solver_setting.critical_radius

    for kappa in np.arange(-1,-(N_partial_waves+1+1),-1,dtype=int):
        if phase_difference_gr0:    
            #print(kappa,'calc')
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
                    if verbose:
                        print("phase differences set to zero after kappa=",kappa)
        else:
            #print(kappa,'0')
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
        print('Warning: m!=0 does not converge properly, to be revised')
        #mass_correction = mass_correction_amplitude(energy,theta,lepton_mass,N_partial_waves,phase_shifts)
        #spinflip = np.tan(theta/2)*nonspinflip + mass_correction
        spinflip = spinflip_amplitude(energy,theta,lepton_mass,N_partial_waves,subtractions,phase_shifts)
        crosssection = np.abs(nonspinflip)**2 + np.abs(spinflip)**2
        
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

def mass_correction_amplitude(energy,theta,lepton_mass,N_partial_waves,phase_shifts):
    # needs subtraction -> TODO
    k=momentum(energy,lepton_mass)
    amplitude=0
    for kappa in np.arange(1,N_partial_waves+1,dtype=int):
        coefficient=np.exp(2j*phase_shifts[kappa])-np.exp(2j*phase_shifts[-kappa])
        amplitude+=coefficient*(kappa*np.tan(theta/2)*associated_legendre(0,kappa-1,np.cos(theta)) - associated_legendre(1,kappa-1,np.cos(theta)))
    return amplitude/(2j*k)
