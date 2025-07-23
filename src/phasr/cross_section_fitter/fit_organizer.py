from ..config import local_paths

from .. import constants

import numpy as np
pi = np.pi

import os

from .fit_performer import fitter
from .initializer import initializer
from .data_prepper import load_dataset

from multiprocessing import Pool, cpu_count

from ..utility.mpsentinel import MPSentinel
MPSentinel.As_master()

def parallel_fitting(datasets_keys:list,Z:int,A:int,Rs=np.arange(5.00,12.00,0.25),N_base_offset=0,N_base_span=2,N_processes=cpu_count()-2,**args):
    
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
            R=Rs[i]
            N=Ns[i]
            for N_offset in np.arange(N-N_base_span,N+N_base_span+1,1,dtype=int):
                if N_offset>1:
                    pairings.append((datasets_keys,Z,A,R,N_offset,args))
        
        # ADD check if this fit is on file 
        
        N_tasks = len(pairings)
        print('Queuing',N_tasks,'tasks, which will be performed over',N_processes,'processes.')
        
        with Pool(processes=np.min([N_processes,N_tasks])) as pool:  # maxtasksperchild=1
            results = pool.starmap(fit_runner,pairings)
        
    return { 'R_'+str(pairings[i][3]) + 'N_'+str(pairings[i][4]) : results[i] for i in range(len(results))}

def fit_runner(datasets_keys,Z,A,R,N,args):
    print("Start fit with R="+str(R)+", N="+str(N)+" (PID:"+str(os.getpid())+")")
    
    # ADD check and load initials from previous fit
    
    initialization = initializer(Z,A,R,N)
    result = fitter(datasets_keys,initialization,**args)
    print("Finished fit with R="+str(R)+", N="+str(N)+" (PID:"+str(os.getpid())+")")
    return result