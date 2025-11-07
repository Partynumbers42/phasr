from ... import constants

from ..continuumstate import continuumstates
from ...nuclei.parameterizations.coulomb import delta_coulomb, eta_coulomb

import numpy as np
pi = np.pi

from scipy.special import lpmv as associated_legendre

from ...utility.math import momentum

import itertools

import time, copy

from ...utility.spliner import save_and_load

from ...config import local_paths

# multiprocessing
from multiprocessing import Pool, cpu_count
# module as master
from ...utility.mpsentinel import MPSentinel
MPSentinel.As_master()

parameter_steps={
    'method' : np.array(['DOP853']), #,'LSODA' 
    'N_partial_waves' : np.append([250,200,150],np.append(np.arange(120,50-10,-10),np.arange(50,20-5,-5))),
    'atol' : 10**np.arange(-13,-6+1,1,dtype=float),
    'rtol' : 10**np.arange(-13,-6+1,1,dtype=float),
    'energy_norm': constants.hc*10**np.arange(0,-6-1,-1,dtype=float),
    'phase_difference_limit' : np.append([0],10**np.arange(-15,-6+1,1,dtype=float)),
}

def crosssection_and_time(energy,theta,nucleus,N_processes=1,**args):
        start_time=time.time()
        crosssection = crosssection_lepton_nucleus_scattering(energy,theta,nucleus,N_processes=N_processes,**args)
        end_time=time.time()
        runtime=end_time-start_time
        return crosssection, runtime

def crosssection_lepton_nucleus_scattering_multithreaded_args(energy,theta,nucleus,lepton_mass=0,recoil=True,subtractions=3,verbose=False,args_dict=parameter_steps,N_max_cpu=cpu_count()-1):

    N_params_iter=len(list(itertools.product(*args_dict.values())))
    N_pools = int(np.min([N_max_cpu,N_params_iter]))
    
    results_dict={}
    crossection_dict={}
    time_dict={}
    
    if MPSentinel.Is_master():
        params_iter = itertools.product(*args_dict.values())
        pool_counter=0
        parameter_counter=0
        for method,N_partial_waves,atol,rtol,energy_norm,phase_difference_limit in params_iter: 
            
            args={'method':method,'N_partial_waves':N_partial_waves,'atol':atol,'rtol':rtol,'energy_norm':energy_norm,'phase_difference_limit':phase_difference_limit,'lepton_mass':lepton_mass,'recoil':recoil,'subtractions':subtractions,'verbose':verbose} 
            
            if pool_counter==0:
                pool = Pool(processes=N_pools)
            
            results_dict[parameter_counter] = pool.apply_async(crosssection_and_time, args=(energy,theta,nucleus), kwds=args)
            print('Queued: ',parameter_counter+1,'/',N_params_iter,' , ',pool_counter+1,'/',N_pools)
            
            pool_counter+=1
            parameter_counter+=1
            
            if pool_counter==N_pools:
                pool_counter=0
                pool.close()
                pool.join()
                
        for key in results_dict:
            crosssection_key, runtime_key =  results_dict[key].get()
            crossection_dict[key] = crosssection_key
            time_dict[key] = runtime_key

    return crossection_dict, time_dict

def optimise_crosssection_precision_multithreaded_args(energy,theta,nucleus,lepton_mass=0,recoil=True,subtractions=3,crosssection_precision=1e-3,jump_forward_dist=1,verbose=False,N_max_cpu=cpu_count()-1):
    
    N_params_iter=len(list(itertools.product(*parameter_steps.values())))
    N_pools = int(np.min([N_max_cpu,N_params_iter]))
    
    results_dict={}
    crosssection_dict={}
    runtime_dict={}
    args_dict={}
    gotten_keys=[]
    
    if MPSentinel.Is_master():
        
        insufficient_args=[]
        
        pool_counter=0
        parameter_counter=0
        
        first=True
        for method,N_partial_waves,atol,rtol,energy_norm,phase_difference_limit in itertools.product(*parameter_steps.values()): 
            
            skip=False
            
            args={'method':method,'N_partial_waves':N_partial_waves,'atol':atol,'rtol':rtol,'energy_norm':energy_norm,'phase_difference_limit':phase_difference_limit} 
            
            for key in args:
                index_key = np.where(parameter_steps[key]==args[key])[0][0]
                if index_key>0:
                    args_check = copy.copy(args)
                    args_check[key] = parameter_steps[key][index_key-1]
                    # skip if more precise calculation was already unsuccessful 
                    if args_check in insufficient_args:
                        skip=True
                        insufficient_args.append(args)
                        #print('skipped')
                        break
                
                if index_key<len(parameter_steps[key]) and not first:
                    index_best_key = np.where(parameter_steps[key]==best_args[key])[0][0]
                    # skip until close to the current best again 
                    if index_key < index_best_key-jump_forward_dist:
                        skip=True
                        #print('jumped forward')
                        break
            
            if not skip:
                
                if pool_counter==0:
                    pool = Pool(processes=N_pools)
            
                args_dict[parameter_counter]=args
                
                results_dict[parameter_counter] = pool.apply_async(crosssection_and_time, args=(energy,theta,nucleus), kwds=args)
                print('Queued: ',parameter_counter+1,'/',N_params_iter,' , ',pool_counter+1,'/',N_pools) 

                pool_counter+=1
                parameter_counter+=1
                
                if pool_counter==N_pools:
                    print('Queued: ',parameter_counter,'/',N_params_iter)
                    pool.close()
                    pool.join()
                    pool_counter=0
                    
                    for key in results_dict:
                        if not (key in gotten_keys):
                            crosssection_key, runtime_key =  results_dict[key].get()
                            crosssection_dict[key] = crosssection_key
                            runtime_dict[key] = runtime_key
                
                    if first:
                        crosssection0=crosssection_dict[0]
                        best_time=runtime_dict[0]
                        best_args=args_dict[0]
                        first=False
                    
                    for key in results_dict:
                        if not (key in gotten_keys):

                            crossections_difference=np.abs(crosssection_dict[key]-crosssection0)/crosssection0
                        
                            if np.all(crossections_difference<crosssection_precision) and (runtime_dict[key]-best_time)/best_time<1e-2:
                                best_time=runtime_dict[key]
                                best_args=args_dict[key]
                                if True:#verbose: 
                                    print('new best:',best_args)
                                    print('time:',best_time,'diff:',np.max(crossections_difference))
                            
                            if np.any(crossections_difference>crosssection_precision):
                                insufficient_args.append(args)
                                
                            gotten_keys.append(key)
            else:
                N_params_iter-=1

    if verbose:
        print('best time:',best_time)    
    
    return best_args

def optimise_crosssection_precision(energy,theta,nucleus,lepton_mass=0,recoil=True,subtractions=3,crosssection_precision=1e-3,jump_forward_dist=1,N_processes=1,verbose=False):

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
                # skip if more precise calculation was already unsuccessful
                if args_check in insufficient_args:
                    skip=True
                    insufficient_args.append(args)
                    #print('skipped')
                    break
            
            if index_key<len(parameter_steps[key]) and not first:
                index_best_key = np.where(parameter_steps[key]==best_args[key])[0][0]
                # skip until close to the current best again
                if index_key < index_best_key-jump_forward_dist:
                    skip=True
                    #print('jumped forward')
                    break

        if not skip:

            crosssection, runtime = crosssection_and_time(energy,theta,nucleus,N_processes=N_processes,lepton_mass=lepton_mass,recoil=recoil,subtractions=subtractions,verbose=verbose,**args)
            
            if first:
                crosssection0=crosssection
                best_time=runtime
                best_args=copy.copy(args)
                first=False
            
            crossections_difference=np.abs(crosssection-crosssection0)/crosssection0
        
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

def phase_shift_from_partial_wave(nucleus,kappa,energy,lepton_mass,**args):
    partial_wave_kappa = continuumstates(nucleus,kappa,energy,lepton_mass,**args)
    partial_wave_kappa.extract_phase_shift()
    return partial_wave_kappa.phase_shift, partial_wave_kappa.phase_difference

def phase_shift_from_partial_wave_wrapper(nucleus,kappa,energy,lepton_mass,save_and_load_phase_shifts=False,verbose=False,**args):
    
    args = {'kappa':kappa,'energy':energy,'lepton_mass':lepton_mass,**args}
    
    path = local_paths.phase_shift_path + 'phase_shift_'+nucleus.name+'_E'+str(energy)+'_kappa'+str(kappa)+'_m'+str(lepton_mass)+'.txt'
    save = save_and_load_phase_shifts
    renew =  not save_and_load_phase_shifts
    return save_and_load(path,renew=renew,save=save,verbose=verbose,fmt='%.50e',fct=phase_shift_from_partial_wave,tracked_params=args,nucleus=nucleus,**args)

def crosssection_lepton_nucleus_scattering(energy,theta,nucleus,lepton_mass=0,recoil=True,subtractions=3,N_partial_waves=250,verbose=False,phase_difference_limit=0,N_processes=1,**args):
    
    nucleus_mass=nucleus.mass
    
    if recoil:
        energy, theta, scale_crosssection = recoil_quantities(energy,theta,nucleus_mass)
        if verbose:
            print('E=',energy,'MeV')
    else:
        scale_crosssection = 1
    
    if N_processes>1:
        phase_shifts, _ = collect_phase_shifts_multithreaded(energy,nucleus,lepton_mass,N_partial_waves,verbose,phase_difference_limit,N_max_cpu=N_processes,**args)
    else:
        phase_shifts, _ = collect_phase_shifts_singlethreaded(energy,nucleus,lepton_mass,N_partial_waves,verbose,phase_difference_limit,**args)
        
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

def collect_phase_shifts_multithreaded(energy,nucleus,lepton_mass,N_partial_waves,verbose,phase_difference_limit,N_max_cpu=cpu_count()-1,**args):
    
    # needs to be improved, does not work most of the time due to local objects
    
    args['verbose']=verbose
    N_pools = int(np.min([N_max_cpu,N_partial_waves]))
    
    charge = nucleus.total_charge
    
    phase_shifts = {}
    phase_differences = {}
    phase_difference_gr0 = True
    
    if MPSentinel.Is_master():
    
        # calculate beginning and critical radius only once, since independent on kappa
        if (not ('beginning_radius' in args)) or (not ('critical_radius' in args)):
            initializer = continuumstates(nucleus,-1,energy,lepton_mass,**args)
        if not 'beginning_radius' in args:
            args['beginning_radius']=initializer.solver_setting.beginning_radius
        if not 'critical_radius' in args:
            args['critical_radius']=initializer.solver_setting.critical_radius

        pool_counter=0
        kappa_counter=0
        kappa=-1
        
        while np.abs(kappa) <= N_partial_waves+1:
        #for kappa in np.arange(-1,-(N_partial_waves+1+1),-1,dtype=int):
            
            if pool_counter==0:
                pool = Pool(processes=N_pools)
                results_dict={}
            
            if phase_difference_gr0:
                
                results_dict[kappa] = pool.apply_async(phase_shift_from_partial_wave_wrapper, args=(nucleus,kappa,energy,lepton_mass), kwds=args)
                #print('Queued: kappa=',kappa,' , (',pool_counter+1,'/',N_pools,')')
                pool_counter+=1
                #phase_shifts[kappa], phase_differences[kappa] = phase_shift_from_partial_wave(nucleus,kappa,energy,lepton_mass,**args) #phase_shift_from_partial_wave(nucleus,kappa,energy,lepton_mass,**args)
                
                if -kappa < N_partial_waves+1:
                    if lepton_mass==0:
                        results_dict[-kappa] = results_dict[kappa]
                    else:
                        results_dict[-kappa] = pool.apply_async(phase_shift_from_partial_wave_wrapper, args=(nucleus,-kappa,energy,lepton_mass), kwds=args)
                        #print('Queued: kappa=',kappa,' , ()',pool_counter+1,'/',N_pools,')')
                        pool_counter+=1
                        #phase_shifts[-kappa], phase_differences[-kappa] = phase_shift_from_partial_wave(nucleus,-kappa,energy,lepton_mass,**args) #phase_shift_from_partial_wave(nucleus,kappa,energy,lepton_mass,**args
                
                kappa_counter+=1
            
                if pool_counter==N_pools:
                    pool_counter=0
                    pool.close()
                    pool.join()
                    
                    for key in results_dict:
                        phase_shift_kappa, phase_difference_kappa =  results_dict[key].get()
                        phase_shifts[key] = phase_shift_kappa
                        phase_differences[key] = phase_difference_kappa

                        if np.abs(phase_differences[key])<=phase_difference_limit:
                            phase_difference_gr0 = False
                            kappa=-np.abs(key)+1
                            if verbose:
                                print("phase differences set to zero after kappa=",kappa)
                            break

            else:
                #print(kappa,'0')
                eta_regular = eta_coulomb(kappa,charge,energy,lepton_mass,reg=+1)
                phase_shifts[kappa] = delta_coulomb(kappa,charge,energy,lepton_mass,reg=+1,pass_eta=eta_regular) + 0
                if -kappa < N_partial_waves+1:
                    if lepton_mass==0:
                        phase_shifts[-kappa] = phase_shifts[kappa]
                    else:
                        phase_shifts[-kappa] = delta_coulomb(-kappa,charge,energy,lepton_mass,reg=+1,pass_eta=eta_regular) + 0
            
            kappa-=1
    
    return phase_shifts, phase_differences

def collect_phase_shifts_singlethreaded(energy,nucleus,lepton_mass,N_partial_waves,verbose,phase_difference_limit,**args):
    
    charge = nucleus.total_charge
    
    phase_shifts = {}
    phase_differences = {}
    phase_difference_gr0 = True
    # calculate beginning and critical radius only once, since independent on kappa
    if (not ('beginning_radius' in args)) or (not ('critical_radius' in args)):
        initializer = continuumstates(nucleus,-1,energy,lepton_mass,verbose=verbose,**args)
    if not 'beginning_radius' in args:
        args['beginning_radius']=initializer.solver_setting.beginning_radius
    if not 'critical_radius' in args:
        args['critical_radius']=initializer.solver_setting.critical_radius

    for kappa in np.arange(-1,-(N_partial_waves+1+1),-1,dtype=int):
        
        if phase_difference_gr0:    
            
            if verbose:
                print('Calculate phaseshift for kappa=',kappa,', ',end="")
                
            phase_shifts[kappa], phase_differences[kappa] = phase_shift_from_partial_wave_wrapper(nucleus,kappa,energy,lepton_mass,verbose=False,**args) #phase_shift_from_partial_wave(nucleus,kappa,energy,lepton_mass,**args)
            
            if verbose:
                print('delta_diff= ',phase_differences[kappa])

            if -kappa < N_partial_waves+1:
                if lepton_mass==0:
                    phase_shifts[-kappa] = phase_shifts[kappa]
                    phase_differences[-kappa] = phase_differences[kappa]
                else:
                    if verbose:
                        print('Calculate phaseshift for kappa=',-kappa,', delta_diff= ',end="")
                    phase_shifts[-kappa], phase_differences[-kappa] = phase_shift_from_partial_wave_wrapper(nucleus,-kappa,energy,lepton_mass,verbose=False,**args) #phase_shift_from_partial_wave(nucleus,kappa,energy,lepton_mass,**args
                    if verbose:
                        print(phase_differences[-kappa])
                if np.abs(phase_differences[kappa])<=phase_difference_limit:
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
    
    return phase_shifts, phase_differences

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
