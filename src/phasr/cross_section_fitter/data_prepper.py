from .. import constants
from .. import trafos

import numpy as np
pi = np.pi

from ..dirac_solvers import crosssection_lepton_nucleus_scattering

def load_data(path,**args): #,correlation_stat_uncertainty=None,correlation_syst_uncertainty=None
    
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
    
    E_data = cross_section_dataset_input[:,E_col]*E_scale
    
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
    
    theta_data = cross_section_dataset_input[:,theta_col]*theta_scale
    
    # Collect cross section
    cross_section_or_fraction = input("Does the file contain direct cross sections or relative measurements to a different nucleus? (answer with: direct or relative)")
    
    if cross_section_or_fraction=="direct":
        
        # Collect cross section directly
        cross_section_col = int(input("What column (starting at 0) contains the central values for the cross section?"))
        cross_section_units = input("In what units is the cross section (fmsq, cmsq, mub, mb or invMeVsq)?")
        if cross_section_units=="fmsq":
            cross_section_scale=1
        elif E_units=="cmsq":
            cross_section_scale=trafos.cmsq_to_fmsq
        elif E_units=="mub":
            cross_section_scale=trafos.mub_to_fmsq
        elif E_units=="mb":
            cross_section_scale=trafos.mb_to_fmsq
        elif E_units=="invMeVsq":
            cross_section_scale=trafos.invMeVsq_to_fmsq
        else:
            cross_section_scale=float(input("Unknown unit. With what factor would I need to multiply these values to transform them to fmsq?"))
        
        cross_section_data = cross_section_dataset_input[:,cross_section_col]*cross_section_scale
        
        stat_vs_syst = input("Does the data distinguish between statistical and systematical uncertainties? (y or n)")
        # Ask for correlations? But how would these be supplied?
        
        if stat_vs_syst == 'y':
            
            # Collect statistical uncertainties
            cross_section_uncertainty_stat_cols = tuple(input("What columns (starting at 0), if any, contain statistical uncertainties for the cross sections (separate by comma)?"))
            if len(cross_section_uncertainty_stat_cols)>0:
                cross_section_uncertainty_stat_abs_or_rel = input("Are the statistical uncertainties absolute values or relative to the central value? (answer with: absolute or relative)")
                
                if cross_section_uncertainty_stat_abs_or_rel == "absolute":
                    cross_section_uncertainty_stat_data = cross_section_dataset_input[:,cross_section_uncertainty_stat_cols]*cross_section_scale
                    
                elif cross_section_uncertainty_stat_abs_or_rel == "relative":
                    cross_section_uncertainty_stat_percent = input("Are the relative statistical uncertainties in percent and thus need to divided by 100? (y or n)")
                    if cross_section_uncertainty_stat_percent == "y":
                        cross_section_uncertainty_stat_scale = 1e-2
                    elif cross_section_uncertainty_stat_percent == "n":
                        cross_section_uncertainty_stat_scale = 1
                        
                    cross_section_uncertainty_stat_data = np.einsum('ij,i->ij',cross_section_dataset_input[:,cross_section_uncertainty_stat_cols]*cross_section_uncertainty_stat_scale,cross_section_data)
                    
            else:
                cross_section_uncertainty_stat_rel_global= float(input("What global relative uncertainty w.r.t. the cross section should instead be considered as a statistical uncertainty (value between 0 and 1, type 0 if you do not want to consider this uncertainty component)?"))
                cross_section_uncertainty_stat_data = cross_section_data*cross_section_uncertainty_stat_rel_global
            
            # Collect systematical uncertainties
            cross_section_uncertainty_syst_cols = tuple(input("What columns (starting at 0), if any, contain systematical uncertainties for the cross sections (separate by comma)?"))
            if len(cross_section_uncertainty_syst_cols)>0:
                cross_section_uncertainty_syst_abs_or_rel = input("Are the systematical uncertainties absolute values or relative to the central value? (answer with: absolute or relative)")
                
                if cross_section_uncertainty_syst_abs_or_rel == "absolute":
                    cross_section_uncertainty_syst_data = cross_section_dataset_input[:,cross_section_uncertainty_syst_cols]*cross_section_scale
                    
                elif cross_section_uncertainty_syst_abs_or_rel == "relative":
                    cross_section_uncertainty_syst_percent = input("Are the relative systematical uncertainties in percent and thus need to divided by 100? (y or n)")
                    if cross_section_uncertainty_syst_percent == "y":
                        cross_section_uncertainty_syst_scale = 1e-2
                    elif cross_section_uncertainty_syst_percent == "n":
                        cross_section_uncertainty_syst_scale = 1
                        
                    cross_section_uncertainty_syst_data = np.einsum('ij,i->ij',cross_section_dataset_input[:,cross_section_uncertainty_syst_cols]*cross_section_uncertainty_syst_scale,cross_section_data)
                    
            else:
                cross_section_uncertainty_syst_rel_global= float(input("What global relative uncertainty w.r.t. the cross section should instead be considered as a systematical uncertainty (value between 0 and 1, type 0 if you do not want to consider this uncertainty component)?"))
                cross_section_uncertainty_syst_data = cross_section_data*cross_section_uncertainty_syst_rel_global
            
        else: 
            # Collect general uncertainties
            cross_section_uncertainty_stat_and_syst_cols = tuple(input("What columns (starting at 0) contain the uncertainties for the cross sections (separate by comma)?"))
            if len(cross_section_uncertainty_stat_and_syst_cols)>0:
                cross_section_uncertainty_stat_and_syst_abs_or_rel = input("Are the uncertainties absolute values or relative to the central value? (answer with: absolute or relative)")
                
                if cross_section_uncertainty_stat_and_syst_abs_or_rel == "absolute":
                    cross_section_uncertainty_stat_and_syst_data = cross_section_dataset_input[:,cross_section_uncertainty_stat_and_syst_cols]*cross_section_scale
                    
                elif cross_section_uncertainty_stat_and_syst_abs_or_rel == "relative":
                    cross_section_uncertainty_stat_and_syst_percent = input("Are the relative systematical uncertainties in percent and thus need to divided by 100? (y or n)")
                    if cross_section_uncertainty_stat_and_syst_percent == "y":
                        cross_section_uncertainty_stat_and_syst_scale = 1e-2
                    elif cross_section_uncertainty_stat_and_syst_percent == "n":
                        cross_section_uncertainty_stat_and_syst_scale = 1
                        
                    cross_section_uncertainty_stat_and_syst_data = np.einsum('ij,i->ij',cross_section_dataset_input[:,cross_section_uncertainty_stat_and_syst_cols]*cross_section_uncertainty_stat_and_syst_scale,cross_section_data)
                
            else:
                cross_section_uncertainty_stat_and_syst_rel_global= input("What percentage of the cross section should instead be considered as a uncertainty (type 0 if you do not want to consider this uncertainty component)?")
                cross_section_uncertainty_stat_and_syst_data = cross_section_data*cross_section_uncertainty_stat_and_syst_rel_global
            
            cross_section_uncertainty_stat_and_syst_split = tuple(input("In what ratio do you want to consider statistical and systematical uncertainty components contributing to the given total uncertainties? (e.g.: (1,1) or (2,1) or (1,0))"))
            
            cross_section_uncertainty_stat_split = np.sqrt(cross_section_uncertainty_stat_and_syst_split[0]/(cross_section_uncertainty_stat_and_syst_split[0]+cross_section_uncertainty_stat_and_syst_split[1]))
            cross_section_uncertainty_syst_split = np.sqrt(cross_section_uncertainty_stat_and_syst_split[1]/(cross_section_uncertainty_stat_and_syst_split[0]+cross_section_uncertainty_stat_and_syst_split[1]))
            
            cross_section_uncertainty_stat_data = cross_section_uncertainty_stat_and_syst_data*cross_section_uncertainty_stat_split
            cross_section_uncertainty_syst_data = cross_section_uncertainty_stat_and_syst_data*cross_section_uncertainty_syst_split
        
        # set correlations
        cross_section_correlation_stat_data = np.identity(N_data)
        cross_section_correlation_syst_data = np.ones((N_data,N_data))
        
    elif cross_section_or_fraction=="relative":

        # Collect 
        Z, N = input("Relative to which nucleus was the data measured? (answer with: Z,N)")
        
        reference_nucleus = SOLUTION_LOADER_TODO(Z,N) #TODO
        
        # if not raise info to first load data for that nucleus
        
        cross_section_reference_data=np.array([])
        for E in np.unique(E_data):
            theta_data_E = theta_data[E_data==E]
            cross_section_reference_data_E = crosssection_lepton_nucleus_scattering(E,theta_data_E,reference_nucleus) # set args for this nucleus 
            cross_section_reference_data = np.append(cross_section_reference_data,cross_section_reference_data_E)
        
        q_data=2*E_data/constants.hc*np.sin(theta_data/2)
        
        form_factor_reference = FF_TODO(q_data,reference_nucleus) #TODO
        form_factor_uncertainty_reference = FF_UNCERT_TODO(q_data,reference_nucleus) #TODO
        form_factor_correlation_reference = FF_CORR_TODO(q_data,reference_nucleus) #TODO
        dcross_section_dform_factor = 2*(cross_section_reference_data/np.abs(form_factor_reference))
        
        cross_section_uncertainty_reference_data = dcross_section_dform_factor * form_factor_uncertainty_reference
        cross_section_correlation_reference_data = form_factor_correlation_reference
        cross_section_covariance_reference_data = np.einsum("i,ij,j->ij",cross_section_uncertainty_reference_data,cross_section_correlation_reference_data,cross_section_uncertainty_reference_data)
        
        # Collect relative cross section measurement
        cross_section_rel_col = int(input("What column (starting at 0) contains the central values for the relative cross section?"))
        cross_section_rel_percent = input("Are the relative cross sections in percent and thus need to divided by 100? (y or n)")
        if cross_section_rel_percent == "y":
            cross_section_rel_scale = 1e-2
        elif cross_section_rel_percent == "n":
            cross_section_rel_scale = 1
        
        cross_section_rel_sign = float(input("If the relative measurement is assumed to be sign*(reference - target)/(reference + target). What value would sign have for your measurement?"))
        cross_section_rel_data = cross_section_rel_sign*cross_section_dataset_input[:,cross_section_rel_col]*cross_section_rel_scale
        
        cross_section_data=cross_section_reference_data * (1.-cross_section_rel_data)/(1.+cross_section_rel_data)
        
        # Collect statistical uncertainties
        cross_section_rel_uncertainty_stat_cols = tuple(input("What columns (starting at 0), if any, contain statistical uncertainties for the relative cross sections (separate by comma)?"))
        if len(cross_section_rel_uncertainty_stat_cols)>0:
            cross_section_rel_uncertainty_stat_data = cross_section_dataset_input[:,cross_section_rel_uncertainty_stat_cols]*cross_section_rel_scale    
        else:
            cross_section_rel_uncertainty_stat_rel_global= float(input("What global relative uncertainty w.r.t. the cross section should instead be considered as a statistical uncertainty (value between 0 and 1, type 0 if you do not want to consider this uncertainty component)?"))
            cross_section_rel_uncertainty_stat_data = cross_section_rel_data*cross_section_rel_uncertainty_stat_rel_global
        
        cross_section_uncertainty_stat_data = cross_section_rel_uncertainty_stat_data*cross_section_reference_data*2/(1+cross_section_rel_data)**2
    
        # Collect systematical uncertainties
        cross_section_rel_uncertainty_syst_cols = tuple(input("What columns (starting at 0), if any, contain systematical uncertainties for the relative cross sections (separate by comma)?"))
        if len(cross_section_rel_uncertainty_syst_cols)>0:
            cross_section_rel_uncertainty_syst_data = cross_section_dataset_input[:,cross_section_rel_uncertainty_syst_cols]*cross_section_rel_scale    
        else:
            cross_section_rel_uncertainty_syst_rel_global= float(input("What global relative uncertainty w.r.t. the relative cross section should instead be considered as a systematical uncertainty (for 3%% input 0.03 here, type 0 if you do not want to consider this uncertainty component)?"))
            cross_section_rel_uncertainty_syst_data = cross_section_rel_data*cross_section_rel_uncertainty_syst_rel_global
        
        cross_section_uncertainty_syst_from_rel = cross_section_rel_uncertainty_syst_data*cross_section_reference_data*2/(1+cross_section_rel_data)**2
        cross_section_uncertainty_syst_data = np.sqrt(cross_section_uncertainty_reference_data**2 + cross_section_uncertainty_syst_from_rel**2)
        
        cross_section_correlation_stat_data = np.identity(N_data)
        cross_section_correlation_syst_from_rel = np.ones((N_data,N_data))
        cross_section_covariance_syst_from_rel = np.einsum("i,ij,j->ij",cross_section_uncertainty_syst_from_rel,cross_section_correlation_syst_from_rel,cross_section_uncertainty_syst_from_rel)
        cross_section_covariance_syst_data = cross_section_covariance_reference_data + cross_section_covariance_syst_from_rel
        cross_section_correlation_syst_data = np.einsum("i,ij,j->ij",1./cross_section_uncertainty_syst_data,cross_section_covariance_syst_data,1./cross_section_uncertainty_syst_data)
        
    else:
        raise ValueError("input is not either direct or relative.")

    cross_section_dataset_for_fit = np.concatenate((E_data,theta_data,cross_section_data,cross_section_uncertainty_stat_data,cross_section_uncertainty_syst_data),axis=-1)
    cross_section_dataset_for_fit = data_sorter(cross_section_dataset_for_fit,(0,1))
    
    # save these to a fixed location for later access
    
    return cross_section_dataset_for_fit, cross_section_correlation_stat_data, cross_section_correlation_syst_data

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
