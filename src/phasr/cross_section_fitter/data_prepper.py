import os

from ..config import local_paths

from .. import constants
from .. import trafos

import numpy as np
pi = np.pi

import glob
#import re

from ..dirac_solvers import crosssection_lepton_nucleus_scattering
from .pickler import load_best_fit
#from ..nuclei import load_reference_nucleus
from .. import nucleus

def import_dataset(path:str,save_name:str,Z:int,A:int,correlation_stat_uncertainty=None,correlation_syst_uncertainty=None,**args):
    
    with open( path, "rb" ) as file:
        cross_section_dataset_input = np.loadtxt( file , **args )
    
    # Collect energies 
    E_col = int(input("What column (starting at 0) contains the central values for the energie?"))
    E_units = input("In what units is the energy (MeV or GeV)?")
    if E_units=="MeV":
        E_scale=1
    elif E_units=="GeV":
        E_scale=1e3
    else:
        E_scale=float(input("Unknown unit. With what factor would I need to multiply these values to transform them to MeV?"))
    
    E_data = cross_section_dataset_input[:,(E_col,)]*E_scale
    
    N_data=len(E_data)
    
    # Collect angles
    theta_col = int(input("What column (starting at 0) contains the central values for the angles?"))
    theta_units = input("In what units is the angle theta (deg or rad)?")
    if theta_units=="deg":
        theta_scale=pi/180
    elif E_units=="rad":
        theta_scale=1
    else:
        theta_scale=float(input("Unknown unit. With what factor would I need to multiply these values to transform them to rad?"))
    
    theta_data = cross_section_dataset_input[:,(theta_col,)]*theta_scale
    
    # Collect cross section
    cross_section_or_fraction = input("Does the file contain direct cross sections or relative measurements to a different nucleus? (answer with: direct or relative)")
    
    if cross_section_or_fraction=="direct":
        
        # Collect cross section directly
        cross_section_col = int(input("What column (starting at 0) contains the central values for the cross section?"))
        cross_section_units = input("In what units is the cross section (fmsq, cmsq, mub, mb or invMeVsq)?")
        if cross_section_units=="fmsq":
            cross_section_scale=1
        elif cross_section_units=="cmsq":
            cross_section_scale=trafos.cmsq_to_fmsq
        elif cross_section_units=="mub":
            cross_section_scale=trafos.mub_to_fmsq
        elif cross_section_units=="mb":
            cross_section_scale=trafos.mb_to_fmsq
        elif cross_section_units=="invMeVsq":
            cross_section_scale=trafos.invMeVsq_to_fmsq
        else:
            cross_section_scale=float(input("Unknown unit. With what factor would I need to multiply these values to transform them to fmsq?"))
        
        cross_section_data = cross_section_dataset_input[:,(cross_section_col,)]*cross_section_scale
        
        stat_vs_syst = input("Does the data distinguish between statistical and systematical uncertainties? (y or n)")
        
        if stat_vs_syst == 'y':
            
            # Collect statistical uncertainties
            cross_section_uncertainty_stat_cols = input("What columns (starting at 0), if any, contain statistical uncertainties for the cross sections (separate by comma)?")
            
            if len(cross_section_uncertainty_stat_cols)>0:
                
                cross_section_uncertainty_stat_cols = tuple(int(col) for col in cross_section_uncertainty_stat_cols.strip("()").split(","))
                
                cross_section_uncertainty_stat_abs_or_rel = input("Are the statistical uncertainties absolute values or relative to the central value? (answer with: absolute or relative)")
                
                if cross_section_uncertainty_stat_abs_or_rel == "absolute":
                    cross_section_uncertainty_stat_data = cross_section_dataset_input[:,cross_section_uncertainty_stat_cols]*cross_section_scale
                    
                elif cross_section_uncertainty_stat_abs_or_rel == "relative":
                    cross_section_uncertainty_stat_percent = input("Are the relative statistical uncertainties in percent and thus need to divided by 100? (y or n)")
                    if cross_section_uncertainty_stat_percent == "y":
                        cross_section_uncertainty_stat_scale = 1e-2
                    elif cross_section_uncertainty_stat_percent == "n":
                        cross_section_uncertainty_stat_scale = 1
                        
                    cross_section_uncertainty_stat_data = np.einsum('ij,i->ij',cross_section_dataset_input[:,cross_section_uncertainty_stat_cols]*cross_section_uncertainty_stat_scale,cross_section_data[:,0])
                    
            else:
                cross_section_uncertainty_stat_data = cross_section_data*0
            
            # Collect systematical uncertainties
            cross_section_uncertainty_syst_cols = input("What columns (starting at 0), if any, contain systematical uncertainties for the cross sections (separate by comma)?")
            if len(cross_section_uncertainty_syst_cols)>0:
                
                cross_section_uncertainty_syst_cols = tuple(int(col) for col in cross_section_uncertainty_syst_cols.strip("()").split(","))
                
                cross_section_uncertainty_syst_abs_or_rel = input("Are the systematical uncertainties absolute values or relative to the central value? (answer with: absolute or relative)")
                
                if cross_section_uncertainty_syst_abs_or_rel == "absolute":
                    cross_section_uncertainty_syst_data = cross_section_dataset_input[:,cross_section_uncertainty_syst_cols]*cross_section_scale
                    
                elif cross_section_uncertainty_syst_abs_or_rel == "relative":
                    cross_section_uncertainty_syst_percent = input("Are the relative systematical uncertainties in percent and thus need to divided by 100? (y or n)")
                    if cross_section_uncertainty_syst_percent == "y":
                        cross_section_uncertainty_syst_scale = 1e-2
                    elif cross_section_uncertainty_syst_percent == "n":
                        cross_section_uncertainty_syst_scale = 1
                        
                    cross_section_uncertainty_syst_data = np.einsum('ij,i->ij',cross_section_dataset_input[:,cross_section_uncertainty_syst_cols]*cross_section_uncertainty_syst_scale,cross_section_data[:,0])
                    
            else:
                cross_section_uncertainty_syst_data = cross_section_data*0
            
        else: 
            # Collect general uncertainties
            cross_section_uncertainty_stat_and_syst_cols = input("What columns (starting at 0) contain the uncertainties for the cross sections (separate by comma)?")
            if len(cross_section_uncertainty_stat_and_syst_cols)>0:
                
                cross_section_uncertainty_stat_and_syst_cols = tuple(int(col) for col in cross_section_uncertainty_stat_and_syst_cols.strip("()").split(","))
                
                cross_section_uncertainty_stat_and_syst_abs_or_rel = input("Are the uncertainties absolute values or relative to the central value? (answer with: absolute or relative)")
                
                if cross_section_uncertainty_stat_and_syst_abs_or_rel == "absolute":
                    cross_section_uncertainty_stat_and_syst_data = cross_section_dataset_input[:,cross_section_uncertainty_stat_and_syst_cols]*cross_section_scale
                    
                elif cross_section_uncertainty_stat_and_syst_abs_or_rel == "relative":
                    cross_section_uncertainty_stat_and_syst_percent = input("Are the relative systematical uncertainties in percent and thus need to divided by 100? (y or n)")
                    if cross_section_uncertainty_stat_and_syst_percent == "y":
                        cross_section_uncertainty_stat_and_syst_scale = 1e-2
                    elif cross_section_uncertainty_stat_and_syst_percent == "n":
                        cross_section_uncertainty_stat_and_syst_scale = 1
                        
                    cross_section_uncertainty_stat_and_syst_data = np.einsum('ij,i->ij',cross_section_dataset_input[:,cross_section_uncertainty_stat_and_syst_cols]*cross_section_uncertainty_stat_and_syst_scale,cross_section_data[:,0])
                
            else:
                cross_section_uncertainty_stat_and_syst_rel_global= input("What percentage of the cross section should instead be considered as a uncertainty (type 0 if you do not want to consider this uncertainty component)?")
                cross_section_uncertainty_stat_and_syst_data = cross_section_data*cross_section_uncertainty_stat_and_syst_rel_global
            
            cross_section_uncertainty_stat_and_syst_split = input("In what ratio do you want to consider statistical and systematical uncertainty components contributing to the given total uncertainties? (e.g.: (1,1) or (2,1) or (1,0))")
            
            cross_section_uncertainty_stat_and_syst_split = tuple(float(split) for split in cross_section_uncertainty_stat_and_syst_split.strip("()").split(","))
            
            cross_section_uncertainty_stat_split = np.sqrt(cross_section_uncertainty_stat_and_syst_split[0]/(cross_section_uncertainty_stat_and_syst_split[0]+cross_section_uncertainty_stat_and_syst_split[1]))
            cross_section_uncertainty_syst_split = np.sqrt(cross_section_uncertainty_stat_and_syst_split[1]/(cross_section_uncertainty_stat_and_syst_split[0]+cross_section_uncertainty_stat_and_syst_split[1]))
            
            cross_section_uncertainty_stat_data = cross_section_uncertainty_stat_and_syst_data*cross_section_uncertainty_stat_split
            cross_section_uncertainty_syst_data = cross_section_uncertainty_stat_and_syst_data*cross_section_uncertainty_syst_split
        
        cross_section_uncertainty_stat_rel_global= float(input("Do you want to add a global relative statistical uncertainty w.r.t. the cross section as an (additional) uncertainty component? (For 3% insert 3, type 0 if you do not want to do so)?"))/100
        cross_section_uncertainty_stat_data = np.sqrt(cross_section_uncertainty_stat_data**2 + (cross_section_data*cross_section_uncertainty_stat_rel_global)**2)
        
        cross_section_uncertainty_syst_rel_global= float(input("Do you want to add a global relative systematical uncertainty w.r.t. the cross section as an (additional) uncertainty component? (For 3% insert 3, type 0 if you do not want to do so)?"))/100
        cross_section_uncertainty_syst_data = np.sqrt(cross_section_uncertainty_syst_data**2 + (cross_section_data*cross_section_uncertainty_syst_rel_global)**2)    
        
        # Set correlations
        if correlation_stat_uncertainty is None:
            cross_section_correlation_stat_data = np.identity(N_data)
        else:
            cross_section_correlation_stat_data = correlation_stat_uncertainty
            
        if correlation_syst_uncertainty is None:
            cross_section_correlation_syst_data = np.ones((N_data,N_data))
        else:
            cross_section_correlation_syst_data = correlation_syst_uncertainty
        
    elif cross_section_or_fraction=="relative":

        print('Warning: import of relative cross sections is an experimental feature')
        # TODO Check that this works in principle 
        
        # Collect 
        Z_ref_str, A_ref_str = input("Relative to which nucleus was the data measured? (answer with: Z,N)").split(',')
        Z_ref, A_ref = int(Z_ref_str), int(A_ref_str)
        
        reference_nucleus_fit_results, _ = load_best_fit(Z_ref,A_ref,verbose=True)
        
        if not reference_nucleus_fit_results is None:
            ai_ref = reference_nucleus_fit_results['ai']
            R_ref = reference_nucleus_fit_results['R']
            reference_nucleus = nucleus(name="reference_nucleus_Z"+str(Z_ref)+"_A"+str(A_ref),Z=Z_ref,A=A_ref,ai=ai_ref,R=R_ref)
            covariance_ai = reference_nucleus_fit_results['cov_ai_model']
        else:
            raise LookupError('Fit results for this nucleus not found. Promote a fit for this nucleus to best fit first.')
        
        cross_section_reference_data=np.array([])
        
        #E_data = np.squeeze(E_data)
        #theta_data = np.squeeze(theta_data)
        
        for E in np.unique(E_data):
            theta_data_E = theta_data[E_data==E]
            cross_section_reference_data_E = crosssection_lepton_nucleus_scattering(E,theta_data_E,reference_nucleus)*constants.hc**2 # set args for this nucleus 
            cross_section_reference_data = np.append(cross_section_reference_data,cross_section_reference_data_E)
        
        q_data=np.squeeze(2*E_data/constants.hc*np.sin(theta_data/2))
        
        form_factor_reference = reference_nucleus.form_factor(q_data)
        form_factor_jacobian = reference_nucleus.form_factor_jacobian(q_data)
        form_factor_covariance_reference = np.einsum("ji,jk,kl->il",form_factor_jacobian,covariance_ai,form_factor_jacobian)
        
        dcross_section_dform_factor = 2*(cross_section_reference_data/np.abs(form_factor_reference))
        cross_section_covariance_reference_data = np.einsum("i,ij,j->ij",dcross_section_dform_factor,form_factor_covariance_reference,dcross_section_dform_factor)
        cross_section_uncertainty_reference_data = np.sqrt(cross_section_covariance_reference_data.diagonal())
        
        cross_section_reference_data = cross_section_reference_data[:,np.newaxis]
        cross_section_uncertainty_reference_data = cross_section_uncertainty_reference_data[:,np.newaxis]
        
        # Collect relative cross section measurement
        cross_section_rel_col = int(input("What column (starting at 0) contains the central values for the relative cross section?"))
        cross_section_rel_percent = input("Are the relative cross sections in percent and thus need to divided by 100? (y or n)")
        if cross_section_rel_percent == "y":
            cross_section_rel_scale = 1e-2
        elif cross_section_rel_percent == "n":
            cross_section_rel_scale = 1
        
        cross_section_rel_sign = float(input("If the relative measurement is assumed to be sign*(reference - target)/(reference + target). What value would sign have for your measurement?"))
        cross_section_rel_data = cross_section_rel_sign*cross_section_dataset_input[:,(cross_section_rel_col,)]*cross_section_rel_scale
        #cross_section_rel_data = np.squeeze(cross_section_rel_data)
        
        cross_section_data=cross_section_reference_data * (1.-cross_section_rel_data)/(1.+cross_section_rel_data)
        
        # Collect statistical uncertainties
        cross_section_rel_uncertainty_stat_cols = input("What columns (starting at 0), if any, contain statistical uncertainties for the relative cross sections (separate by comma)?")
        if len(cross_section_rel_uncertainty_stat_cols)>0:
            cross_section_rel_uncertainty_stat_cols = tuple(int(col) for col in cross_section_rel_uncertainty_stat_cols.strip("()").split(","))
            cross_section_rel_uncertainty_stat_data = cross_section_dataset_input[:,cross_section_rel_uncertainty_stat_cols]*cross_section_rel_scale
            #cross_section_rel_uncertainty_stat_data = np.squeeze(cross_section_rel_uncertainty_stat_data)    
        else:
            cross_section_rel_uncertainty_stat_rel_global= float(input("What global relative uncertainty w.r.t. the cross section should instead be considered as a statistical uncertainty (value between 0 and 1, type 0 if you do not want to consider this uncertainty component)?"))
            cross_section_rel_uncertainty_stat_data = cross_section_rel_data*cross_section_rel_uncertainty_stat_rel_global
        
        cross_section_uncertainty_stat_data = cross_section_rel_uncertainty_stat_data*cross_section_reference_data*2/(1+cross_section_rel_data)**2
        
        # Collect systematical uncertainties
        cross_section_rel_uncertainty_syst_cols = input("What columns (starting at 0), if any, contain systematical uncertainties for the relative cross sections (separate by comma)?")
        if len(cross_section_rel_uncertainty_syst_cols)>0:
            cross_section_rel_uncertainty_syst_cols = tuple(int(col) for col in cross_section_rel_uncertainty_syst_cols.strip("()").split(","))
            cross_section_rel_uncertainty_syst_data = cross_section_dataset_input[:,cross_section_rel_uncertainty_syst_cols]*cross_section_rel_scale
            #cross_section_rel_uncertainty_syst_data = np.squeeze(cross_section_rel_uncertainty_syst_data)
        else:
            cross_section_rel_uncertainty_syst_rel_global= float(input("What global relative uncertainty w.r.t. the relative cross section should instead be considered as a systematical uncertainty (for 3%% input 0.03 here, type 0 if you do not want to consider this uncertainty component)?"))
            cross_section_rel_uncertainty_syst_data = cross_section_rel_data*cross_section_rel_uncertainty_syst_rel_global
        
        cross_section_uncertainty_syst_from_rel = cross_section_rel_uncertainty_syst_data*cross_section_reference_data*2/(1+cross_section_rel_data)**2
        cross_section_uncertainty_syst_data = np.sqrt(cross_section_uncertainty_reference_data**2 + cross_section_uncertainty_syst_from_rel**2)
        
        # Set correlations
        if correlation_stat_uncertainty is None:
            cross_section_correlation_stat_data = np.identity(N_data)
        else:
            cross_section_correlation_stat_data = correlation_stat_uncertainty
        
        if correlation_syst_uncertainty is None:
            cross_section_correlation_syst_from_rel = np.ones((N_data,N_data))
        else:
            cross_section_correlation_syst_from_rel = correlation_syst_uncertainty
        
        # Add model uncertainties
        cross_section_covariance_syst_from_rel = np.einsum("i,ij,j->ij",np.squeeze(cross_section_uncertainty_syst_from_rel),cross_section_correlation_syst_from_rel,np.squeeze(cross_section_uncertainty_syst_from_rel))
        cross_section_covariance_syst_data = cross_section_covariance_reference_data + cross_section_covariance_syst_from_rel
        cross_section_correlation_syst_data = np.einsum("i,ij,j->ij",1./np.squeeze(cross_section_uncertainty_syst_data),cross_section_covariance_syst_data,1./np.squeeze(cross_section_uncertainty_syst_data))
        
    else:
        raise ValueError("input is not either direct or relative.")

    # more squeeze to application
    cross_section_dataset_for_fit = np.concatenate((E_data,theta_data,cross_section_data,cross_section_uncertainty_stat_data,cross_section_uncertainty_syst_data),axis=-1)
    #cross_section_dataset_for_fit = data_sorter(cross_section_dataset_for_fit,(0,1))
    
    dataset_name = save_name + "_Z"+str(Z) + "_A"+str(A)
    save_path_cross_section = local_paths.cross_section_data_path + "cross_section_" + dataset_name + ".txt"
    save_path_cross_section_correlation_stat = local_paths.cross_section_data_path + "cross_section_" + dataset_name + "_correlation_stat.txt"
    save_path_cross_section_correlation_syst = local_paths.cross_section_data_path + "cross_section_" + dataset_name + "_correlation_syst.txt"
    
    os.makedirs(os.path.dirname(save_path_cross_section), exist_ok=True)
    
    with open(save_path_cross_section, "wb" ) as file:
        np.savetxt(file,cross_section_dataset_for_fit)
        print("cross section data saved in ", save_path_cross_section)
    
    with open(save_path_cross_section_correlation_stat, "wb" ) as file:
        np.savetxt(file,cross_section_correlation_stat_data)
        print("cross section statistical correlation data saved in ", save_path_cross_section_correlation_stat)
    
    with open(save_path_cross_section_correlation_syst, "wb" ) as file:
        np.savetxt(file,cross_section_correlation_syst_data)
        print("cross section systematical correlation data saved in ", save_path_cross_section_correlation_syst)
    
    print("The dataset "+dataset_name+" can now be accessed by the fitting routines")


def list_datasets(Z,A):
    path_beginning = local_paths.cross_section_data_path + "cross_section_"
    path_ending = "_Z"+str(Z) + "_A"+str(A) + ".txt"
    path_pattern = path_beginning + "*" + path_ending
    existing_paths = glob.glob(path_pattern)
    existing_datasets =[path[len(path_beginning):-len(path_ending)] for path in existing_paths]
    print("These loaded datasets were found for Z="+str(Z)+" and A="+str(A)+":")
    print(existing_datasets)

def load_dataset(name,Z,A,verbose=True):
    
    dataset_name = name + "_Z"+str(Z) + "_A"+str(A)
    save_path_cross_section = local_paths.cross_section_data_path + "cross_section_" + dataset_name + ".txt"
    save_path_cross_section_correlation_stat = local_paths.cross_section_data_path + "cross_section_" + dataset_name + "_correlation_stat.txt"
    save_path_cross_section_correlation_syst = local_paths.cross_section_data_path + "cross_section_" + dataset_name + "_correlation_syst.txt"
    
    with open(save_path_cross_section, "rb" ) as file:
        cross_section_dataset_for_fit = np.loadtxt(file, dtype=float)
        if verbose:
            print("cross section data loaded from ", save_path_cross_section)
    
    with open(save_path_cross_section_correlation_stat, "rb" ) as file:
        cross_section_correlation_stat_data = np.loadtxt(file, dtype=float)
        if verbose:
            print("cross section statistical correlation data loaded from ", save_path_cross_section_correlation_stat)
    
    with open(save_path_cross_section_correlation_syst, "rb" ) as file:
        cross_section_correlation_syst_data = np.loadtxt(file, dtype=float)
        if verbose:
            print("cross section systematical correlation data loaded from ", save_path_cross_section_correlation_syst)
    
    return cross_section_dataset_for_fit, cross_section_correlation_stat_data, cross_section_correlation_syst_data


def import_barrett_moment(name:str,Z:int,A:int,k:float,alpha:float,barrett:float,dbarrett:float):
    
    barrett_data = np.array([('k',k),('alpha',alpha),('barrett',barrett),('dbarrett',dbarrett)],dtype=[('var','<U10'),('val','<f8')])
    
    dataset_name = name + "_Z"+str(Z) + "_A"+str(A)
    save_path_barrett = local_paths.barrett_moment_data_path + "barrett_moment_" + dataset_name + ".txt"
    
    os.makedirs(os.path.dirname(save_path_barrett), exist_ok=True)
    
    with open(save_path_barrett, "wb" ) as file:
        np.savetxt(file,barrett_data,fmt="%s %.4f")
        print("barrett moment value saved in ", save_path_barrett)
    
    print("The barrett moment value with label "+dataset_name+" can now be accessed by the fitting routines")

def list_barrett_moments(Z,A):
    path_beginning = local_paths.barrett_moment_data_path + "barrett_moment_"
    path_ending = "_Z"+str(Z) + "_A"+str(A) + ".txt"
    path_pattern = path_beginning + "*" + path_ending
    existing_paths = glob.glob(path_pattern)
    existing_datasets =[path[len(path_beginning):-len(path_ending)] for path in existing_paths]
    print("These loaded datasets were found for Z="+str(Z)+" and A="+str(A)+":")
    print(existing_datasets)

def load_barrett_moment(name,Z,A,verbose=True):
    
    dataset_name = name + "_Z"+str(Z) + "_A"+str(A)
    save_path_barrett = local_paths.barrett_moment_data_path + "barrett_moment_" + dataset_name + ".txt"
    
    with open(save_path_barrett, "rb" ) as file:
        barrett_moment_data = np.genfromtxt( file,names=None,dtype=['<U10',float])       
        if verbose:
            print("barrett moment value loaded from ", save_path_barrett)
    
    barrett_moment_dict = {str(barrett_moment_data['f0'][i]):float(barrett_moment_data['f1'][i]) for i in range(4)}
    
    return barrett_moment_dict

def data_sorter(data,sort_cols=(0,)):
    transformed_data=np.copy(data)
    initial=True
    for col in sort_cols[::-1]:
        if initial:
            transformed_data=transformed_data[transformed_data[:,col].argsort(),:]
            initial=False
        else:
            transformed_data=transformed_data[transformed_data[:,col].argsort(kind='mergesort'),:]
    return transformed_data
