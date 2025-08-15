import os

import numpy as np

from scipy.interpolate import splev, splrep
from ..config import local_paths

import glob
import re

from functools import partial

def calc_and_spline(fct,xrange,name,dtype=float,ext=0,renew=False,save=True,verbose=True):
    # xrange is always real, fct(x) can be complex
    
    x_str = 'x='+str(xrange[0])+'-'+str(xrange[1])+'-'+str(xrange[2])
    path = local_paths.spline_path + name + "_" + x_str + ".txt"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if os.path.exists(path) and (renew==False or fct is None):
        with open( path, "rb" ) as file:
            xy_data = np.loadtxt( file , dtype=dtype)
            x_data = xy_data[:,0]
            y_data = xy_data[:,1]
        if verbose:
            print("data loaded from ",path)

    else:

        if fct is None:
            raise NameError('data path not found and no fct given to generate')

        if verbose:
            print("data not found at "+path+" or forced to recreate.\nThis may take some time.")

        x_data = np.arange(xrange[0], xrange[1], xrange[2], dtype=dtype)

        y_data = fct(x_data)

        if save:
            with open( path, "wb" ) as file:
                xy_data=np.stack([x_data,y_data],axis=-1)
                np.savetxt(file,xy_data)
                if verbose:
                    print("data saved in ", path)

    if dtype==complex:
        y_data_spl_re = splrep(np.real(x_data),np.real(y_data),s=0)
        y_data_spl_im = splrep(np.real(x_data),np.imag(y_data),s=0)

        fkt_spl = partial(fkt_spl_complex,y_data_spl_re=y_data_spl_re,y_data_spl_im=y_data_spl_im,ext=ext)
        
    elif dtype==float:
        y_data_spl = splrep(x_data,y_data,s=0)
        fkt_spl = partial(fkt_spl_real,y_data_spl=y_data_spl,ext=ext)

    return fkt_spl

def fkt_spl_complex(x,y_data_spl_re,y_data_spl_im,ext):
    if np.imag(x)!=0:
        raise ValueError("complex spline only valid for real values of x")
    return splev(np.real(x),y_data_spl_re,ext=ext) + 1j*splev(np.real(x),y_data_spl_im,ext=ext)

def fkt_spl_real(x,y_data_spl,ext):
    return splev(x,y_data_spl,ext=ext)


def save_and_load(path,renew=False,save=True,verbose=True,fmt='%.18e',fct=None,tracked_params={},**params):
    
    if save or (not renew):
        postfix_name='_args'
        
        params_str=str(tracked_params)
        
        suffix_search = re.search(r'\.\w*$', path)
        if suffix_search is not None:
            suffix = suffix_search.group()
            path = path[:-len(suffix)]
        else:
            suffix = '.txt'
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        existing_paths = glob.glob(path+"*"+suffix)
        
        # check what files are there 
        data_found=False
        del_paths=[]
        taken_postfix_nrs=[]
        for path_i in existing_paths:
            with open( path_i, "rb" ) as file:
                header_line = file.readline().decode('utf-8')
                used_params_str = header_line[2:-10]
                
                if used_params_str==params_str:
                    if renew:
                        del_paths+=[path_i]
                    else:
                        data_structure = header_line[-6:-1]
                        data = np.loadtxt( file , dtype=float)
                        if data_structure == 'scalar':
                            data=data[0]
                        if verbose:
                            print("data loaded from ",path_i)
                        data_found=True
                        break
                else:
                    postfix_nr = int(re.search(postfix_name+r'\d*$', path_i[:-len(suffix)]).group()[len(postfix_name):])
                    taken_postfix_nrs.append(postfix_nr)
            
        # delete old files if they should be renewed 
        for del_path in del_paths:
            if verbose:
                print("deleted "+del_path+"("+params_str+") to renew calculation")
            os.remove(del_path)
        
        if len(taken_postfix_nrs)>0:
            new_postfix_nr=np.max(taken_postfix_nrs)+1
        else:
            new_postfix_nr=0
        postfix = postfix_name+str(new_postfix_nr)
    else:
        data_found=False
        
    # renew files if they are not there
    if not data_found:
        if fct is None:
            raise ValueError("no data to load at ",path+suffix)
        if verbose:
            print("data not found or forced to recalculate.\nThis may take some time.")
        data = fct(**params)
        data_arr = np.atleast_1d(data)
        if save:
            
            path = path + postfix + suffix
            
            with open( path, "wb" ) as file:
                np.savetxt(file,data_arr,fmt=fmt,header=str(tracked_params)+' (scalar)' if np.isscalar(data) else str(tracked_params)+' (array) ')
                if verbose:
                    print("data saved in ", path)
    
    return data
