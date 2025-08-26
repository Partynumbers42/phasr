from ... import constants

import numpy as np
pi = np.pi

import itertools

import time, copy

from functools import partial

# multiprocessing
from multiprocessing import Pool, cpu_count
# module as master
from ...utility.mpsentinel import MPSentinel
MPSentinel.As_master()

from .crosssection import crosssection_lepton_nucleus_scattering

from ...utility.math import momentum_transfer

parameter_steps={
    'method' : np.array(['DOP853']), #,'LSODA' 
    'N_partial_waves' : np.append([250,200,150],np.append(np.arange(120,50-10,-10),np.arange(50,20-5,-5))),
    'atol' : 10**np.arange(-13,-6+1,1,dtype=float),
    'rtol' : 10**np.arange(-13,-6+1,1,dtype=float),
    'energy_norm': constants.hc*10**np.arange(0,-6-1,-1,dtype=float),
    'phase_difference_limit' : np.append([0],10**np.arange(-15,-6+1,1,dtype=float)),
}

def crosssection_lepton_nucleus_scattering_chirality(energy,theta,chirality,weak_nucleus,charge_nucleus=None,verbose=False,**args):
    
    if charge_nucleus is None:
        charge_nucleus = weak_nucleus

    args['verbose']=verbose
    
    if chirality=='L':
        chiral_sign=-1
        chirality_name='left'
    elif chirality=='R':
        chiral_sign=+1
        chirality_name='right'
    else:
        raise NameError("Only two chiralities: 'L' , 'R'")
    
    if verbose:
        print('Calculate '+chirality_name+' crosssection ...')

    potential_chiral = partial(custom_chiral_potential,Vch=charge_nucleus.electric_potential,Vw=weak_nucleus.weak_potential,sign=chiral_sign)
    
    nucleus_chiral = copy.deepcopy(charge_nucleus)
    nucleus_chiral.name += charge_nucleus.name+'_'+weak_nucleus.name+'_'+chirality
    nucleus_chiral.electric_potential = potential_chiral
    return crosssection_lepton_nucleus_scattering(energy,theta,nucleus_chiral,**args)

def custom_chiral_potential(r,Vch,Vw,sign):
    return Vch(r) + sign*Vw(r)

def left_right_asymmetry_lepton_nucleus_scattering(energy,theta,weak_nucleus,charge_nucleus=None,verbose=False,parallelize_LR=False,acceptance=None,atol=1e-13,rtol=1e-13,**args):
    
    if charge_nucleus is None:
        charge_nucleus = weak_nucleus

    args['verbose']=verbose
    args['atol']=atol
    args['rtol']=rtol
    
    if parallelize_LR:
        # left and right are run in parallel
        results_dict={}
        if MPSentinel.Is_master():
            pool = Pool(processes=2)
            for chirality in ['L','R']:
                results_dict[chirality] = pool.apply_async(crosssection_lepton_nucleus_scattering_chirality, args=(energy,theta,chirality,weak_nucleus,charge_nucleus), kwds=args)
            pool.close()
            pool.join()
        crosssection_L = results_dict['L'].get()
        crosssection_R = results_dict['R'].get()
    else:
        crosssection_L = crosssection_lepton_nucleus_scattering_chirality(energy,theta,'L',weak_nucleus,charge_nucleus,**args)
        crosssection_R = crosssection_lepton_nucleus_scattering_chirality(energy,theta,'R',weak_nucleus,charge_nucleus,**args)
    
    left_right_asymmetry = (crosssection_R - crosssection_L)/(crosssection_R + crosssection_L)
    
    if acceptance is None:
    
        return left_right_asymmetry
    
    else:
        
        left_right_asymmetry_weighted_mean_L = acceptance_weight(left_right_asymmetry,theta,acceptance,crosssection_L)
        theta_weighted_mean_L = acceptance_weight(theta,theta,acceptance,crosssection_L)
        Qsq_weighted_mean_L = acceptance_weight(momentum_transfer(energy,theta,charge_nucleus.mass)**2,theta,acceptance,crosssection_L)
        
        left_right_asymmetry_weighted_mean_R = acceptance_weight(left_right_asymmetry,theta,acceptance,crosssection_R)
        theta_weighted_mean_R = acceptance_weight(theta,theta,acceptance,crosssection_R)
        Qsq_weighted_mean_R = acceptance_weight(momentum_transfer(energy,theta,charge_nucleus.mass)**2,theta,acceptance,crosssection_R)
        
        left_right_asymmetry_weighted_mean = np.mean([left_right_asymmetry_weighted_mean_L,left_right_asymmetry_weighted_mean_R])
        theta_weighted_mean = np.mean([theta_weighted_mean_L,theta_weighted_mean_R])
        Qsq_weighted_mean = np.mean([Qsq_weighted_mean_L,Qsq_weighted_mean_R])
        
        return theta_weighted_mean, Qsq_weighted_mean, left_right_asymmetry_weighted_mean

def acceptance_weight(quantity,theta,acceptance,crosssection):
    return np.sum(acceptance*np.sin(theta)*crosssection*quantity) / np.sum(acceptance*np.sin(theta)*crosssection)    

def left_right_asymmetry_and_time(energy,theta,nucleus,*args,**kwds):
        start_time=time.time()
        LR_asymmetry = left_right_asymmetry_lepton_nucleus_scattering(energy,theta,nucleus,*args,**kwds)
        end_time=time.time()
        runtime=end_time-start_time
        return LR_asymmetry, runtime

def optimise_left_right_asymmetry_precision(energy,theta,nucleus,left_right_asymmetry_precision=1e-3,jump_forward_dist=1,N_processes=1,verbose=False):

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

            LR_asymmetry, runtime = left_right_asymmetry_and_time(energy,theta,nucleus,N_processes=N_processes,verbose=verbose,**args)
            
            if first:
                LR_asymmetry0=LR_asymmetry
                best_time=runtime
                best_args=copy.copy(args)
                first=False
            
            LR_asymmetry_difference=np.abs(LR_asymmetry-LR_asymmetry0)/LR_asymmetry0
        
            if np.all(LR_asymmetry_difference<left_right_asymmetry_precision) and (runtime-best_time)/best_time<1e-2:
                if True:#verbose:
                    print('new best:',args)
                    print('time:',runtime,'diff:',np.max(LR_asymmetry_difference))
                best_time=runtime
                best_args=copy.copy(args)
            
            if np.any(LR_asymmetry_difference>left_right_asymmetry_precision):
                insufficient_args.append(args)

    if verbose:
        print('best time:',best_time)    
    
    return best_args