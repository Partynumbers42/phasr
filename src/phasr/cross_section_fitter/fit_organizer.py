from ..config import local_paths

from .. import constants

import numpy as np
pi = np.pi

import os
import copy

from .fit_performer import fitter
from .fit_initializer import initializer
from .data_prepper import load_dataset
from .. import nucleus

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

def fit_runner(datasets_keys,Z,A,R,N,args):
    print("Start fit with R="+str(R)+", N="+str(N)+" (PID:"+str(os.getpid())+")")
    
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

def select_RN_based_on_property(results_dict,property,limit,sign=+1):
    
    RN_tuples=[]
    for key in results_dict:
        if sign*results_dict[key][property] > sign*limit:
            RN_tuples.append((results_dict[key]['R'],results_dict[key]['N']))
    
    return RN_tuples

def split_based_on_asymptotic_and_p_val(results_dict,qs=[400,580,680,1000],dq=1.,m=None,p_val_lim=0):
    
    if m is None:
        q0=np.arange(qs[0],qs[1],dq)
        q1=np.arange(qs[1],qs[2],dq)
        q2=np.arange(qs[2],qs[3],dq)
        As=[]
        ms=[]
        for key in results_dict:
            #
            result = results_dict[key]
            nuc_result = nucleus('temp_nuc_'+key,Z=result['Z'],A=result['A'],ai=result['ai'],R=result['R'])
            #
            F0=np.abs(nuc_result.form_factor(q0))
            F1=np.abs(nuc_result.form_factor(q1))
            F0_max=np.max(F0)
            q0_max=q0[np.argmax(F0)]
            F1_max=np.max(F1)
            q1_max=q1[np.argmax(F1)]
            #
            m_key=-np.log(F1_max/F0_max)/np.log(q1_max/q0_max)
            #
            A=F0_max*q0_max**m_key
            As.append(A)
            ms.append(m_key)
        m_min=np.min(ms)
        A=As[np.argmin(ms)]
    else:
        m_min=0
        
    if m_min<4:# or (m is not None):
        if m is None:
            m_min= 4
            q0=np.arange(qs[1],qs[2],dq)
            q2=np.arange(qs[2],qs[3],dq)
        else:
            m_min= m
            q0=np.arange(qs[0],qs[1],dq)
            q2=np.arange(qs[1],qs[2],dq)
            
        As=[]
        for key in results_dict:
            #
            result = results_dict[key]
            nuc_result = nucleus('temp_nuc_'+key,Z=result['Z'],A=result['A'],ai=result['ai'],R=result['R'])
            #
            F0=np.abs(nuc_result.form_factor(q0))
            F0_max=np.max(F0)
            q0_max=q0[np.argmax(F0)]
            #
            A=F0_max*q0_max**m_min
            As.append(A)
        A=np.max(As)
    
    print('Asymptotic Parameter: m='+str(m_min))
    
    F2_lim = A/q2**m_min
    
    def limit_asymptotic(q):
        return A/q**m_min
    
    results_dict_survive, results_dict_veto = {}, {}
    
    for key in results_dict:
        
        result = results_dict[key]
        nuc_result = nucleus('temp_nuc_'+key,Z=result['Z'],A=result['A'],ai=result['ai'],R=result['R'])
            
        F2=np.abs(nuc_result.form_factor(q2))
        
        if np.all(F2_lim>F2) and result['p_val']>=p_val_lim:
            results_dict_survive[key]=result
        else:
            results_dict_veto[key]=result
        
    return results_dict_survive, results_dict_veto, limit_asymptotic