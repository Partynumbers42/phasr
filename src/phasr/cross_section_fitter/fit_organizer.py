from ..config import local_paths

from .. import constants

import numpy as np
pi = np.pi

import os
import copy

from .fit_performer import fitter
from .fit_initializer import initializer
from .data_prepper import load_dataset

from multiprocessing import Pool, cpu_count

from ..utility.mpsentinel import MPSentinel
MPSentinel.As_master()

def parallel_fitting_manual(datasets_keys:list,Z:int,A:int,RN_tuples=[],N_processes=cpu_count()-2,**args):
    results={}
    
    if MPSentinel.Is_master():    
        
        pairings = []
        
        for i in range(len(RN_tuples)):
            R,N=RN_tuples[i]
            R = np.float64(R)
            N = np.int64(N)
            pairings.append((datasets_keys,Z,A,R,N,args))
        
        N_tasks = len(pairings)
        N_processes = np.min([N_processes,N_tasks])
        
        print('Queuing',N_tasks,'tasks, which will be performed over',N_processes,'processes.')
        
        with Pool(processes=N_processes) as pool:  # maxtasksperchild=1
            results = pool.starmap(fit_runner,pairings)
    
    return { 'R'+str(pairings[i][3]) + '_N'+str(pairings[i][4]) : results[i] for i in range(len(results))}

def parallel_fitting_automatic(datasets_keys:list,Z:int,A:int,Rs=np.arange(5.00,12.00,0.25),N_base_offset=0,N_base_span=2,N_processes=cpu_count()-2,**args):
    
    results={}
    
    if MPSentinel.Is_master():    
        q_max=0
        for dataset_key in datasets_keys:
            dataset, _, _ = load_dataset(dataset_key,Z,A,verbose=False) 
            energy = dataset[:,0]
            theta = dataset[:,1]
            q_mom_approx = 2*energy/constants.hc*np.sin(theta/2)
            q_mom_approx = np.append(q_mom_approx,q_max)
            q_max = np.max(q_mom_approx)
            Ns = np.ceil((Rs*q_max)/pi).astype(int)+N_base_offset
        
        pairings = []
        
        for i in range(len(Rs)):
            R=np.float64(Rs[i])
            N=np.int64(Ns[i])
            for N_offset in np.arange(N-N_base_span,N+N_base_span+1,1,dtype=int):
                if N_offset>2:
                    pairings.append((datasets_keys,Z,A,R,N_offset,args))
        
        N_tasks = len(pairings)
        print('Queuing',N_tasks,'tasks, which will be performed over',N_processes,'processes.')
        
        #print(pairings)
        
        with Pool(processes=np.min([N_processes,N_tasks])) as pool:  # maxtasksperchild=1
            results = pool.starmap(fit_runner,pairings)
    
        
    return { 'R'+str(pairings[i][3]) + '_N'+str(pairings[i][4]) : results[i] for i in range(len(results))}

def select_RN_based_on_property(results_dict,property,limit,sign=+1):
    
    RN_tuples=[]
    for key in results_dict:
        if sign*results_dict[key][property] > sign*limit:
            RN_tuples.append((results_dict[key]['R'],results_dict[key]['N']))
    
    return RN_tuples

def fit_runner(datasets_keys,Z,A,R,N,args):
    print("Start fit with R="+str(R)+", N="+str(N)+" (PID:"+str(os.getpid())+")")
    
    #if 'barrett_moment_keys' in args:
    #    barrett_moment_keys = args['barrett_moment_keys']
    #else:
    #    barrett_moment_keys = []
    #
    #if 'monotonous_decrease_precision' in args:
    #    monotonous_decrease_precision = args['monotonous_decrease_precision']
    #else:
    #    monotonous_decrease_precision = np.inf
    #base_settings = {'datasets':datasets_keys,'datasets_barrett_moment':barrett_moment_keys,'monotonous_decrease_precision':monotonous_decrease_precision}
    
    args = copy.deepcopy(args) # prevents that 'initialize_from' is poped from the source
    
    if 'initialize_from' in args:
        initialize_from = args['initialize_from']
        args.pop('initialize_from')
    else:
        initialize_from = 'reference'
    
    initialization = initializer(Z,A,R,N,initialize_from=initialize_from)
    result = fitter(datasets_keys,initialization,**args)
    print("Finished fit with R="+str(R)+", N="+str(N)+" (PID:"+str(os.getpid())+")")
    return result

    