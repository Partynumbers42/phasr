from ..config import local_paths

import numpy as np

import os
import pickle
import glob
import hashlib

def tracking_str_generator(results_dict,tracked_keys=None,visible_keys=[]):
    
    if tracked_keys is None:
        tracked_keys = list(results_dict.keys())
    
    tracking_str=""
    for key in visible_keys:
        value=results_dict[key]
        if type(value) in [list,np.ndarray]: 
            tracking_str+="_"+key
            for val in value:
                tracking_str+=str(val)
        else:
            tracking_str+="_"+key+str(value)
    
    tracked_dict = {key: results_dict[key] for key in tracked_keys}
    if len(tracked_dict)>0:
        hash=hashlib.sha256()
        tracked_dict_str=str(tracked_dict)
        hash.update(tracked_dict_str.encode(('utf-8')))
        tracking_str+="_"+hash.hexdigest()
        
    return tracking_str
    
def pickle_dump_result_dict(results_dict,tracked_keys=None,visible_keys=[],overwrite=True,verbose=True):
    
    tracking_str = tracking_str_generator(results_dict,tracked_keys,visible_keys)
    path = local_paths.fit_path + 'fit_result' + tracking_str + '.pkl'

    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if (not os.path.exists(path)) or overwrite:
        with open( path, "wb" ) as file:
            pickle.dump(results_dict, file)
        if verbose:
            print('Saving results to',path)
    else:
        print('File at',path,'already exists')

def pickle_load_result_dict(test_dict,tracked_keys=None,visible_keys=[],verbose=True):
    
    tracking_str = tracking_str_generator(test_dict,tracked_keys,visible_keys)
    path = local_paths.fit_path + 'fit_result' + tracking_str + '.pkl'
    
    if os.path.exists(path):
        with open( path, "rb" ) as file:
            results_dict = pickle.load(file) 
        if verbose:
            print('Loading results from',path)
        return results_dict
    else:
        print('File at',path,'not found')

def pickle_load_all_results_dicts_R_N(Z,A,R,N):
        
    test_dict={'Z':Z,'A':A,'R':R,'N':N}
    visible_keys = ['Z','A','R','N']
    
    tracking_str = tracking_str_generator(test_dict,tracked_keys=[],visible_keys=visible_keys)
    path_pattern = local_paths.fit_path + 'fit_result' + tracking_str + '*.pkl'
    #print(path_pattern)
    paths = glob.glob(path_pattern)
    #print(paths)
    
    results_dicts = {}
    
    for path in paths:
        with open( path, "rb" ) as file:
            results_dict = pickle.load(file) 
        results_dicts[os.path.basename(path[:-4])] = results_dict
    
    return results_dicts 

def pickle_load_all_results_dicts_R(Z,A,R,settings):
    # settings = {'datasets':datasets_keys,'datasets_barrett_moment':barrett_moment_keys,'monotonous_decrease_precision':monotonous_decrease_precision,'xi_diff_convergence_limit':xi_diff_convergence_limit,'numdifftools_step':numdifftools_step,**cross_section_args,**minimizer_args}
    
    test_dict={'Z':Z,'A':A,'R':R}
    visible_keys = ['Z','A','R']
    
    tracking_str = tracking_str_generator(test_dict,tracked_keys=[],visible_keys=visible_keys)
    
    path_pattern = local_paths.fit_path + 'fit_result' + tracking_str + '*.pkl'
    #print('path pattern:',path_pattern)
    paths = glob.glob(path_pattern)
    #print('paths:',paths)
    
    results_dicts = {}
    
    for path in paths:
        with open( path, "rb" ) as file:
            results_dict = pickle.load(file) 
        
        same_kwds = True
        for key in settings:
            if results_dict[key] != settings[key]:
                same_kwds=False
        
        if same_kwds:
            results_dicts[os.path.basename(path[:-4])] = results_dict
    
    return results_dicts 

def promote_best_fit(results_dict,overwrite=True,verbose=True):
    
    # ADD syst uncertainties here at some point <------ ?
    
    if not results_dict is None:   
        path = local_paths.best_fit_path + 'best_fit_result_Z' + str(results_dict['Z']) + '_A'  + str(results_dict['A']) + '.pkl'

        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if (not os.path.exists(path)) or overwrite:
            with open( path, "wb" ) as file:
                pickle.dump(results_dict, file)
            if verbose:
                print('Saving results to',path)
        else:
            print('File at',path,'already exists')
        
    
def load_best_fit(Z,A,verbose=True):
    
    path = local_paths.best_fit_path + 'best_fit_result_Z' + str(Z) + '_A'  + str(A) + '.pkl'
    
    if os.path.exists(path):
        with open( path, "rb" ) as file:
            results_dict = pickle.load(file) 
        if verbose:
            print('Loading results from',path)
        return results_dict
    else:
        print('File at',path,'not found')
    

def add_syst_uncertainties(): #TODO 
    pass
