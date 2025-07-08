from .. import constants
from .. import trafos

import numpy as np
pi = np.pi


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
            
            
            
        
    elif cross_section_or_fraction=="relative":
        raise ValueError("Not implemented yet")
    
    else:
    
        raise ValueError("input is not either direct or relative")

    cross_section_dataset_for_fit = np.concatenate((E_data,theta_data,cross_section_data,cross_section_uncertainty_stat_data,cross_section_uncertainty_syst_data),axis=-1)
    cross_section_dataset_for_fit = data_sorter(cross_section_dataset_for_fit,(0,1))
    
    return cross_section_dataset_for_fit

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

