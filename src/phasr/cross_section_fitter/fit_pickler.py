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
    path = local_paths.fit_path + 'fit_result' + tracking_str

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
    path = local_paths.fit_path + 'fit_result' + tracking_str
    
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
    path_pattern = local_paths.fit_path + 'fit_result' + tracking_str+'*.pkl'
    paths = glob.glob(path_pattern)
    
    results_dicts = {}
    
    for path in paths:
        with open( path, "rb" ) as file:
            results_dict = pickle.load(file) 
        results_dicts[os.path.basename(path)] = results_dict
    
    return results_dicts 

def promote_best_fit(test_dict,tracked_keys=None,visible_keys=[],overwrite=True,verbose=True):
    
    results_dict = pickle_load_result_dict(test_dict,tracked_keys,visible_keys,verbose)
    
    if not results_dict is None:   
        path = local_paths.best_fit_path + 'best_fit_result_Z' + str(results_dict['Z']) + '_A'  + str(results_dict['A'])

        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if (not os.path.exists(path)) or overwrite:
            with open( path, "wb" ) as file:
                pickle.dump(results_dict, file)
            if verbose:
                print('Saving results to',path)
        else:
            print('File at',path,'already exists')
        
    
def load_best_fit(Z,A,verbose=True):
    
    path = local_paths.best_fit_path + 'best_fit_result_Z' + str(Z) + '_A'  + str(A)
    
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





# # OLD remove in future

# def path_gen(params_dict,txtname,folder_path='./lib/fits',label_afix=""):
#     str_header=write_header_underscore(params_dict)
#     path=folder_path+"\\"+txtname+str_header+("_"+label_afix if len(label_afix)>0 else "")+'.pkl'
#     return path

# def pickle_withparameters(dict_generator,params_dict,txtname,folder_path='./lib/fits',label_afix="fitset",verbose=True,renew=False,save=True):
    
#     if (not save) and (not renew):
#         return dict_generator(**params_dict)
    
#     header=write_header(params_dict)
    
#     path=path_gen(params_dict,txtname,folder_path,label_afix)
#     #str_header=write_header_underscore(params_dict)
#     #path=folder_path+"\\"+txtname+str_header+("_"+label_afix if len(label_afix)>0 else "")+'.pkl'
    
#     folder_path=os.path.dirname(path)
#     data_found=False
    
#     if os.path.exists(path) and (not renew):
#         with open( path, "rb" ) as file:
#             data = pickle.load(file) 
#             data_found=True
#             if verbose:
#                 print("loaded from"+path+"("+header+")")    
#     #
#     if not data_found:
#         data=dict_generator(**params_dict)
#         if save:
#             # delete old files if they exist
#             if os.path.exists(path):
#                 if verbose:
#                     print("deleted "+path+"("+header+") to renew calculation")
#                 os.remove(path)
#             with open( path, "wb" ) as file:
#                 pickle.dump(data, file)
#             if verbose:
#                 print("saved to"+path+"("+header+")")
    
#     return data


# def saveload_withparameters(data_generator,params_dict,txtname,folder_path='./lib/splines',label_afix="dataset",verbose=True,renew=False,save=True):
    
#     if (not save) and (not renew):
#         return data_generator(**params_dict)
    
#     header=write_header(params_dict)
#     path=folder_path+"\\"+txtname+"_"+label_afix
#     folder_path=os.path.dirname(path)
#     data_found=False
#     paths = glob.glob(path+"*.txt")
#     del_paths=[]
#     for path_i in paths:
#         with open( path_i, "rb" ) as file:
#             header_i=file.readline().decode("utf-8")[2:-1]
#             if header_i==header:
#                 if renew:
#                     del_paths+=[path_i]
#                 else:
#                     data = np.loadtxt( file , dtype=float)
#                     data_found=True
#                     if verbose:
#                         print("loaded from"+path_i+"("+header+")")
#                     break
#     #
#     # delete old files if they should be renewed 
#     for del_path in del_paths:
#         if verbose:
#             print("deleted "+del_path+"("+header+") to renew calculation")
#         os.remove(del_path)
#     #
#     if not data_found:
#         data=data_generator(**params_dict)
#         if save:            
#             i=0
#             while (path+str(i)+".txt" in paths):
#                 i+=1
#             path_i=path+str(i)+".txt"
#             with open( path_i, "wb" ) as file:
#                 np.savetxt(file,data,header=header,fmt='%.50e')
#             if verbose:
#                 print("saved to"+path_i+"("+header+")")
    
#     return data

# def write_header(params_dict):
#     header=""
#     for key in params_dict:
#         param=params_dict[key]
#         if type(param)==np.ndarray:
#             param_hash=hashlib.sha256()
#             paramstring=str(param)
#             param_hash.update(paramstring.encode(('utf-8')))
#             header+=key+" in "+str([np.min(param),np.max(param)])+" (hash:"+param_hash.hexdigest()+"), "
#         elif type(param)==atom.atom:
#             header+=key+"="+param.name+"-Z"+str(param.Z)+"-N"+str(param.N)+"-R"+str(param.R)+"-a"+str(np.mean(param.ai))+", "
#         else:
#             header+=key+"="+str(param)+", "
#     return header[:-2]

# def write_header_underscore(params_dict):
#     header=""
#     for key in params_dict:
#         param=params_dict[key]
#         if type(param)==np.ndarray:
#             param_hash=hashlib.sha256()
#             paramstring=str(param)
#             param_hash.update(paramstring.encode(('utf-8')))
#             header+="_"+key+"-"+param_hash.hexdigest()
#         elif type(param)==atom.atom:
#             header+="_"+key+param.name+"-Z"+str(param.Z)+"-Na"+str(param.N)+"-R"+str(param.R)+"-a"+str(np.mean(param.ai))
#         else:
#             header+="_"+key+str(param)
#     return header