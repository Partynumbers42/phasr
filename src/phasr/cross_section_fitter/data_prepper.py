from .. import constants
from .. import trafos

import numpy as np
pi = np.pi


def load_data(path):
    
    with open( path, "rb" ) as file:
        CS_data = np.loadtxt( file )
        
    E_col = input("What column (starting at 0) contains the central values for the energie?")
    E_units = input("In what units is the energy (MeV or GeV)?")
    if E_units=="MeV":
        E_scale=1
    elif E_units=="GeV":
        E_scale=1e3
    else:
        E_scale=float(input("Unknown unit. With what factor would I need to multiply these values to transform them to MeV?"))
    
    theta_col = input("What column (starting at 0) contains the central values for the angles?")
    theta_units = input("In what units is the angle theta (deg or rad)?")
    if theta_units=="deg":
        theta_scale=pi/180
    elif E_units=="rad":
        theta_scale=1
    else:
        theta_scale=float(input("Unknown unit. With what factor would I need to multiply these values to transform them to rad?"))
    
    cross_section_or_fraction = input("Does the file contain direct cross sections or relative measurements to a different nucleus? (answer with: direct or relative)")
    
    if cross_section_or_fraction=="direct":
        
        cross_section_col = input("What column (starting at 0) contains the central values for the cross section?")
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
        
        
        stat_vs_syst = input("Does the data distinguish between statistical and systematical uncertainties? (y or n)")        
        
        cross_section_uncertainty_stat_cols = input("What columns (starting at 0) contain the uncertainties for the cross sections (separate by comma)?")
        cross_section_uncertainty_syst_cols = input("What columns (starting at 0) contain the uncertainties for the cross sections (separate by comma)?")
        cross_section_units = input("In what units is the cross section (fmsq, cmsq, mub, mb, invMeVsq)?")
        
        
    elif cross_section_or_fraction=="relative":
        pass
    
    
    
    
    else:
    
    
        raise ValueError("input is not either direct or relative")
    
    
    
    
    ,dE_cols=(1,),theta_cols=(2,),dtheta_cols=(2,),cross_section_cols=(3,),dstat_cols=(4,),dsyst_cols=(5,),corr_stat=None,corr_syst=None
    
    
    
    
    pass