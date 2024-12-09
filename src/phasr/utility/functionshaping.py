#!/usr/bin/env python
# coding: utf-8

import numpy as np
#import scipy as sc
pi=np.pi

import hashlib

import pickle

import os.path
import glob
from scipy.interpolate import splev, splrep
from scipy.interpolate import bisplev
#from scipy.interpolate import griddata 
from scipy.interpolate import RectBivariateSpline

import nucliProperties as atom

def continueer(data,jump=2*pi, tresh=pi):

    diffs=np.insert(np.diff(data),0,0)
    diffs_true1 = diffs>tresh

    if diffs_true1.any():

        diff_index1=np.min(np.argwhere(diffs_true1))
        diffs_true1[diff_index1:].fill(True)
        newdata = np.copy(np.where(diffs_true1,data - jump,data))
        newdata=continueer(newdata,jump,tresh)
        data=np.copy(newdata)
        diffs=np.insert(np.diff(data),0,0)

    diffs_true2 = diffs<-tresh


    if diffs_true2.any():

        diff_index2=np.min(np.argwhere(diffs_true2))
        diffs_true2[diff_index2:].fill(True)
        newdata = np.copy(np.where(diffs_true2,data + jump,data))
        newdata=continueer(newdata,jump,tresh)
        data=np.copy(newdata)
        diffs=np.insert(np.diff(data),0,0)

    return data


def continueer2(data,jump=2*pi, tresh=pi):

    diffs=np.insert(np.diff(data),0,0)

    jumpplacesUp=[]
    jumpplacesDown=[]

    diffs_true1 = diffs>tresh

    if diffs_true1.any():
        diff_index1=np.min(np.argwhere(diffs_true1))
        jumpplacesUp.append(diff_index1) # store places
        diffs_true1[diff_index1:].fill(True)
        newdata = np.copy(np.where(diffs_true1,data - jump,data))
        newdata, ups, downs=continueer2(newdata,jump,tresh)
        data=np.copy(newdata)
        jumpplacesUp+=ups
        jumpplacesDown+=downs
        diffs=np.insert(np.diff(data),0,0)

    diffs_true2 = diffs<-tresh

    if diffs_true2.any():
        diff_index2=np.min(np.argwhere(diffs_true2))
        jumpplacesDown.append(diff_index2) # store places
        diffs_true2[diff_index2:].fill(True)
        newdata = np.copy(np.where(diffs_true2,data + jump,data))
        newdata, ups, downs=continueer2(newdata,jump,tresh)
        data=np.copy(newdata)
        jumpplacesUp+=ups
        jumpplacesDown+=downs
        diffs=np.insert(np.diff(data),0,0)

    return data, jumpplacesUp, jumpplacesDown

def renew_phase(phase,sArray,jump=2*pi,tresh=pi,dtype=complex):
    data, ups, downs = continueer2(phase(sArray),jump,tresh)

    newphase = lambda x: phase(x) + sum(np.where(x<sArray[down],0,jump) for down in downs) + sum(np.where(x<sArray[up],0,-jump) for up in ups)
    return newphase, data


def calcandspline2D(fkt,xrange,path,dtype=complex,phase=False,x_GeVtoMeV=False,ext=0,renew=False,save=True,verbose=True):

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

        if phase:
            y_data, ups, downs = continueer2(y_data,jump=2*pi,tresh=pi)

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
        fkt_spl = lambda x: splev(np.real(x),y_data_spl_re,ext=ext) + 1j*splev(np.real(x),y_data_spl_im,ext=ext)

    elif dtype==float:
        y_data_spl = splrep(x_data,y_data,s=0)
        fkt_spl = lambda x: splev(x,y_data_spl,ext=ext)

    return fkt_spl

#import time

def calcandspline3D(fkt,xrange,yrange,path,dtype=complex,renew=False,save=True,verbose=True,return_tck=False):

    #for dtype complex evaluate only on real axis (so x and y should be real, f(x,y) can be complex of course)
    os.makedirs(os.path.dirname(path), exist_ok=True) #makes folders if necessary
    
    if os.path.exists(path) and (renew==False or fkt is None):
        with open( path, "rb" ) as file:
            shape_str = str(file.readline())
            
            shape_strs=shape_str[5:-4].split(', ')
            shape=(int(shape_strs[0]),int(shape_strs[1]),int(shape_strs[2])) # read initial shape from header
            xyz_data_reshaped = np.loadtxt( file , dtype=dtype)
            
            reshape=xyz_data_reshaped.shape
            xyz_data = xyz_data_reshaped.reshape(reshape[0],reshape[1] // shape[2], shape[2]) # reshape back into original shape
            x_data_mesh = xyz_data[:,:,0]
            y_data_mesh = xyz_data[:,:,1]
            x_data=x_data_mesh[0]
            y_data=y_data_mesh[:,0]
            z_data = xyz_data[:,:,2]
        if verbose:
            print("data loaded from ",path)

    else:

        if fkt is None:
            raise NameError('data path not found and no fct given to generate')
            fkt = lambda x: x

        if verbose:
            print("data not found at "+path+" or forced to recreate.\nThis may take some time.")

        x_data = np.arange(xrange[0], xrange[1], xrange[2], dtype=dtype)
        y_data = np.arange(yrange[0], yrange[1], yrange[2], dtype=dtype)
        
        x_data_mesh, y_data_mesh = np.meshgrid(x_data,y_data)
        
        z_data = fkt(x_data_mesh,y_data_mesh)
        
        if save:
            with open( path, "wb" ) as file:
                xyz_data=np.stack([x_data_mesh,y_data_mesh,z_data],axis=-1)
                shape=xyz_data.shape
                xyz_data_reshaped=xyz_data.reshape(shape[0], -1) # reshape to 2D, save initial shape in header
                np.savetxt(file,xyz_data_reshaped,header=str(shape)) 
                if verbose:
                    print("data saved in ", path)
    
    if dtype==complex:
        
        z_data_spl_re = RectBivariateSpline(np.real(x_data),np.real(y_data),np.real(z_data.transpose()))
        z_data_spl_im = RectBivariateSpline(np.real(x_data),np.real(y_data),np.imag(z_data.transpose()))
        #
        if return_tck:
            c_re =z_data_spl_re.get_coeffs()
            tx_re,ty_re=z_data_spl_re.get_knots()
            tck_re = [tx_re, ty_re, c_re, 3, 3]
            #
            c_im =z_data_spl_im.get_coeffs()
            tx_im,ty_im=z_data_spl_im.get_knots()
            tck_im = [tx_im, ty_im, c_im, 3, 3]
            return tck_re, tck_im
        
        # real part of x is taken without checking  im=0!!!!
        fkt_spl = lambda x,y: z_data_spl_re.ev(np.real(x),np.real(y)) + 1j*z_data_spl_im.ev(np.real(x),np.real(y))
        
    elif dtype==float:
        
        z_data_spl = RectBivariateSpline(x_data,y_data,z_data.transpose())#,s=0
        if return_tck:
            c=z_data_spl.get_coeffs()
            tx,ty=z_data_spl.get_knots()
            tck=[tx, ty, c, 3, 3]
            #
            return tck
            
        fkt_spl = lambda x,y: z_data_spl.ev(x,y)
        
        
    return fkt_spl

def bispl_reconstr(tcks,dtype=complex):
    # warning returns x*y array 
    
    if dtype == complex:
        tck_re, tck_im = tcks
        fkt_spl = lambda x,y: bisplev(np.real(x),np.real(y),tck_re) + 1j*bisplev(np.real(x),np.real(y),tck_im)
    else:
        fkt_spl = lambda x,y: bisplev(np.real(x),np.real(y),tcks)
        
    fkt_spl_out = np.vectorize(fkt_spl) # not ideal, but best for now
    
    # def fkt_spl_out_0(x,y): #2D x,y work only if mesh grid shape and x and y are rising
        
        # dimx=False
        # dimy=False
        
        # if np.size(x)>1:
            # #print("x",x.shape)
            # if x.ndim==1:
                # dimx=True
            # elif x.ndim==2:
                # #print("x[0]",x[0].shape)
                # x=x[0]
            # elif x.ndim>2:
                # raise ValueError("x has wrong dimensions")
        # if np.size(y)>1:
            # #print("y",y.shape)
            # if y.ndim==1:
                # dimy=True
            # elif y.ndim==2:
                # #print("y[:,0]",y[:,0].shape)
                # y=y[:,0]
            # elif y.ndim>2:
                # raise ValueError("y has wrong dimensions")
        
        # if dimx ^ dimy:
            # raise ValueError("x and y dim do not match") 
        
        # if np.size(x)>1 and np.size(y)>1:
            # #print("T")
            # if dimx:
                # return np.diag(fkt_spl(x,y))
            # else:
                # return fkt_spl(x,y).transpose()
        # else:
            # return fkt_spl(x,y)
    
    return fkt_spl_out

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


def lowest_file_finder(val_extracter,params_dict,folder_path,suffix=''):
    str_header=write_header_underscore(params_dict)
    paths = glob.glob(folder_path+'*'+str_header+'*'+suffix)
    val_min=np.inf
    path_min=None
    data_min=None
    for path in paths:
        with open( path, "rb" ) as file:
            data = pickle.load(file)
        val = val_extracter(data)
        if val<val_min:
            val_min=val
            path_min=path
            data_min=data
    if path_min is not None:
        print('old/wrong fit found: Starting from: '+path_min)
    return data_min, path_min

##Highenergycontinuations

def Tslope(t,val_crit,deriv_crit,limit):
    """belongs to highenergycontinuation"""
    return ((limit-val_crit)/deriv_crit)*np.exp(-t)

# goes like e^(-s/T)
def highenergycontinuation(s,scrit,val_crit,deriv_crit,limit,t=0):
    """ goes like e^(-s/t) """
    T = Tslope(t,val_crit,deriv_crit,limit)
    phase = limit + (val_crit-limit + (deriv_crit + (val_crit-limit)/Tslope(t,val_crit,deriv_crit,limit))*(s-scrit))*np.exp(-(s-scrit)/T)
    return phase

# goes like (1/s)^n
def highenergycontinuation2(s,scrit,val_crit,deriv_crit,limit,n=1):
    """ goes like (1/s)^n """
    return limit + (limit-val_crit)/((scrit-((s**n)/(scrit**(n-1))))*(deriv_crit/(n*(limit-val_crit))) - 1)

# goes/diverges like log(s)
def highenergycontinuation3(s,scrit,val_crit,deriv_crit):
    """ goes/diverges like log(s) """
    return val_crit + deriv_crit*scrit*np.log(s/scrit)

def highenergycontinuation4_old(s,scrit,val_crit,deriv_crit,A):
    """ goes/oszillates like A*sin(s) """
    q = deriv_crit/np.sqrt(A**2-val_crit**2)
    s0 = np.arcsin(val_crit/A)/q - scrit
    return A*np.sin(q*(s+s0))

def highenergycontinuation4(s,scrit,val_crit,deriv_crit,q,return_params=False):
    """ goes/oszillates like A*sin(q*s + phi) """
    s0 = np.arctan2(val_crit*q,deriv_crit)/q - scrit
    A = val_crit/np.sin(q*(scrit+s0))
    if return_params:
        return A, s0*q
    else:
        return A*np.sin(q*(s+s0))
    #else:
    #    raise ValueError('derv_crit=0')
