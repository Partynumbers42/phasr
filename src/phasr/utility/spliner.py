import os
import numpy as np

from scipy.interpolate import splev, splrep
from ..config import local_paths

def calcandspline(fct,xrange,name,dtype=float,ext=0,renew=False,save=True,verbose=True):
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

        def fkt_spl(x):
            if np.imag(x)!=0:
                raise ValueError("complex spline only valid for real values of x")
            return splev(np.real(x),y_data_spl_re,ext=ext) + 1j*splev(np.real(x),y_data_spl_im,ext=ext)

    elif dtype==float:
        y_data_spl = splrep(x_data,y_data,s=0)
        def fkt_spl(x): return splev(x,y_data_spl,ext=ext)

    return fkt_spl

def saveandload(path,renew=False,save=True,verbose=True,fmt='%.18e',fct=None,**params):
    
    if os.path.exists(path) and renew==False:
        with open( path, "rb" ) as file:
            data = np.loadtxt( file , dtype=float)
            if verbose:
                print("data loaded from ",path)
    elif fct is None:
        raise ValueError("no data to load at ",path)
    else:
        if verbose:
            print("data not found or forced to recalculate.\nThis may take some time.")
        data = fct(**params)
        if save:
            with open( path, "wb" ) as file:
                np.savetxt(file,data,fmt=fmt)
                if verbose:
                    print("data saved in ", path)
    return data