import os

import numpy as np

import pickle
import glob
import hashlib

# --> TODO Rework unused atm

def path_gen(params_dict,txtname,folder_path='./lib/fits',label_afix=""):
    str_header=write_header_underscore(params_dict)
    path=folder_path+"\\"+txtname+str_header+("_"+label_afix if len(label_afix)>0 else "")+'.pkl'
    return path

def pickle_withparameters(dict_generator,params_dict,txtname,folder_path='./lib/fits',label_afix="fitset",verbose=True,renew=False,save=True):
    
    if (not save) and (not renew):
        return dict_generator(**params_dict)
    
    header=write_header(params_dict)
    
    path=path_gen(params_dict,txtname,folder_path,label_afix)
    #str_header=write_header_underscore(params_dict)
    #path=folder_path+"\\"+txtname+str_header+("_"+label_afix if len(label_afix)>0 else "")+'.pkl'
    
    folder_path=os.path.dirname(path)
    data_found=False
    
    if os.path.exists(path) and (not renew):
        with open( path, "rb" ) as file:
            data = pickle.load(file) 
            data_found=True
            if verbose:
                print("loaded from"+path+"("+header+")")    
    #
    if not data_found:
        data=dict_generator(**params_dict)
        if save:
            # delete old files if they exist
            if os.path.exists(path):
                if verbose:
                    print("deleted "+path+"("+header+") to renew calculation")
                os.remove(path)
            with open( path, "wb" ) as file:
                pickle.dump(data, file)
            if verbose:
                print("saved to"+path+"("+header+")")
    
    return data


def saveload_withparameters(data_generator,params_dict,txtname,folder_path='./lib/splines',label_afix="dataset",verbose=True,renew=False,save=True):
    
    if (not save) and (not renew):
        return data_generator(**params_dict)
    
    header=write_header(params_dict)
    path=folder_path+"\\"+txtname+"_"+label_afix
    folder_path=os.path.dirname(path)
    data_found=False
    paths = glob.glob(path+"*.txt")
    del_paths=[]
    for path_i in paths:
        with open( path_i, "rb" ) as file:
            header_i=file.readline().decode("utf-8")[2:-1]
            if header_i==header:
                if renew:
                    del_paths+=[path_i]
                else:
                    data = np.loadtxt( file , dtype=float)
                    data_found=True
                    if verbose:
                        print("loaded from"+path_i+"("+header+")")
                    break
    #
    # delete old files if they should be renewed 
    for del_path in del_paths:
        if verbose:
            print("deleted "+del_path+"("+header+") to renew calculation")
        os.remove(del_path)
    #
    if not data_found:
        data=data_generator(**params_dict)
        if save:            
            i=0
            while (path+str(i)+".txt" in paths):
                i+=1
            path_i=path+str(i)+".txt"
            with open( path_i, "wb" ) as file:
                np.savetxt(file,data,header=header,fmt='%.50e')
            if verbose:
                print("saved to"+path_i+"("+header+")")
    
    return data

def write_header(params_dict):
    header=""
    for key in params_dict:
        param=params_dict[key]
        if type(param)==np.ndarray:
            param_hash=hashlib.sha256()
            paramstring=str(param)
            param_hash.update(paramstring.encode(('utf-8')))
            header+=key+" in "+str([np.min(param),np.max(param)])+" (hash:"+param_hash.hexdigest()+"), "
        elif type(param)==atom.atom:
            header+=key+"="+param.name+"-Z"+str(param.Z)+"-N"+str(param.N)+"-R"+str(param.R)+"-a"+str(np.mean(param.ai))+", "
        else:
            header+=key+"="+str(param)+", "
    return header[:-2]

def write_header_underscore(params_dict):
    header=""
    for key in params_dict:
        param=params_dict[key]
        if type(param)==np.ndarray:
            param_hash=hashlib.sha256()
            paramstring=str(param)
            param_hash.update(paramstring.encode(('utf-8')))
            header+="_"+key+"-"+param_hash.hexdigest()
        elif type(param)==atom.atom:
            header+="_"+key+param.name+"-Z"+str(param.Z)+"-Na"+str(param.N)+"-R"+str(param.R)+"-a"+str(np.mean(param.ai))
        else:
            header+="_"+key+str(param)
    return header