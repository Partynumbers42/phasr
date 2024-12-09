import os
import numpy as np

from scipy.interpolate import splev, splrep

# TODO <- cleanup

def calcandspline2D(fkt,xrange,path,dtype=complex,x_GeVtoMeV=False,ext=0,renew=False,save=True,verbose=True):

    #for dtype complex evaluate only on real axis (so x should be real, f(x) can be complex of course)
    os.makedirs(os.path.dirname(path), exist_ok=True) #makes folders if necessary
    
    if os.path.exists(path) and (renew==False or fkt is None):
        with open( path, "rb" ) as file:
            xy_data = np.loadtxt( file , dtype=dtype)
            #print(xy_data)
            x_data = xy_data[:,0]
            if x_GeVtoMeV:
                x_data*=1e3
            y_data = xy_data[:,1]
        if verbose:
            print("data loaded from ",path)

    else:

        if fkt is None:
            raise NameError('data path not found and no fct given to generate')
            fkt = lambda x: x

        if verbose:
            print("data not found at "+path+" or forced to recreate.\nThis may take some time.")

        x_data = np.arange(xrange[0], xrange[1], xrange[2], dtype=dtype)

        y_data = fkt(x_data)

        if save:
            with open( path, "wb" ) as file:
                xy_data=np.stack([x_data,y_data],axis=-1)
                np.savetxt(file,xy_data)
                if verbose:
                    print("data saved in ", path)

    if dtype==complex:
        y_data_spl_re = splrep(np.real(x_data),np.real(y_data),s=0)
        y_data_spl_im = splrep(np.real(x_data),np.imag(y_data),s=0)

        # real part of x is taken without checking  im=0!!!!
        def fkt_spl(x): return splev(np.real(x),y_data_spl_re,ext=ext) + 1j*splev(np.real(x),y_data_spl_im,ext=ext)

    elif dtype==float:
        y_data_spl = splrep(x_data,y_data,s=0)
        def fkt_spl(x): return splev(x,y_data_spl,ext=ext)

    return fkt_spl