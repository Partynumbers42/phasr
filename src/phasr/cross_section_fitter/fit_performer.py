from ..config import local_paths

from .. import constants
from .. import trafos

import numpy as np
pi = np.pi

from scipy.optimize import minimize
from scipy.integrate import quad

from .statistical_measures import minimization_measures
from .initializer import initializer
from .parameters import parameter_set
from .data_prepper import load_dataset, load_barrett_moment

from ..dirac_solvers import crosssection_lepton_nucleus_scattering

def fitter_single(datasets_keys:list,initialization:initializer,barrett_moment_keys=[],monotonous_decrease_precision=np.inf,convergence_limit=1e-4,cross_section_args={},**minimizer_args):
    ''' **args is passed to scipy minimize '''
    
    initial_parameters = parameter_set(initialization.R,initialization.Z,ai=initialization.ai,ai_abs_bound=initialization.ai_abs_bound)
    current_nucleus = initial_parameters.nucleus # COPY HERE??? #TODO
    
    datasets = {}
    barrett_moment_constraint = (len(barrett_moment_keys)>0)
    monotonous_decrease_constraint = (monotonous_decrease_precision<np.inf)

    for dataset_key in datasets_keys:
        datasets[dataset_key] = {}
        dataset, corr_stat, corr_syst = load_dataset(dataset_key,initialization.Z,initialization.A,verbose=False)    
        dy_stat, dy_syst = dataset[:,3], dataset[:,4]
        datasets[dataset_key]['x_data'] = dataset[:,(0,1)]
        datasets[dataset_key]['y_data'] = dataset[:,2]
        datasets[dataset_key]['cov_stat'] = np.einsum('i,ij,j->ij',dy_stat,corr_stat,dy_stat)
        datasets[dataset_key]['cov_syst'] = np.einsum('i,ij,j->ij',dy_syst,corr_syst,dy_syst)
    
    def cross_section(energy_and_theta,nucleus):
        energy = energy_and_theta[:,0]
        theta = energy_and_theta[:,1]
        cross_section = np.zeros(len(energy))
        for E_val in np.unique(energy):
            mask = (E_val==energy)
            cross_section[mask] = crosssection_lepton_nucleus_scattering(energy,theta[mask],nucleus,**cross_section_args)
    
    if barrett_moment_constraint:
        barrett_moments = {}
        for barrett_moment_key in barrett_moment_keys:
            barrett_moments[barrett_moment_key]={}
            barrett_dict = load_barrett_moment(barrett_moment_key,initialization.Z,initialization.A,verbose=False)
            barrett_moments[barrett_moment_key]['x_data'] = np.nan
            barrett_moments[barrett_moment_key]['y_data'] = barrett_dict["barrett"]
            barrett_moments[barrett_moment_key]['cov_stat'] = barrett_dict["dbarrett"]**2
            barrett_moments[barrett_moment_key]['cov_syst'] = 0
            current_nucleus.update_k_and_alpha_barrett(barrett_dict["k"],barrett_dict["alpha"])
        
        def barrett_moment(_,nucleus):
            return np.atleast_1d(nucleus.barrett_moment)
    
    if monotonous_decrease_constraint:
        def positive_slope_component_to_radius_squared(_,nucleus):
            integrand = lambda r: -4*pi*r**5/5*nucleus.dcharge_density_dr(r)/nucleus.Z
            integrand_positive_slope = lambda r: np.where(integrand(r)<0,integrand(r),0) 
            positive_slope_component = quad(integrand_positive_slope,0,nucleus.R,limit=1000)[0]
            return np.atleast_1d(positive_slope_component)
    
    def loss_function(xi):
    
        parameters = parameter_set(initialization.R,initialization.Z,xi=xi,ai_abs_bound=initialization.ai_abs_bound)
        current_nucleus.update_ai(parameters.get_ai())
        
        measures={}
        for dataset_key in datasets:
            measures[dataset_key]=minimization_measures(cross_section,**datasets[dataset_key])
            
        if barrett_moment_constraint:
            for barrett_moment_key in barrett_moment_keys:
                measures['barrett_moment']=minimization_measures(barrett_moment,**barrett_moments[barrett_moment_key])
        
        if monotonous_decrease_constraint:
            # default: monotonous_decrease_precision=0.04
            measures['monotonous_decrease']=minimization_measures(positive_slope_component_to_radius_squared,x_data=np.nan,y_data=0,cov_stat=monotonous_decrease_precision**2,cov_syst=0)
        
        loss=0
        for dataset_key in measures:
            loss += measures[dataset_key].loss(current_nucleus)

        return loss

    xi_initial = initial_parameters.get_xi()
    xi_bounds = len(xi_initial)*[(0,1)]
    result = minimize(loss_function,xi_initial,bounds=xi_bounds,**minimizer_args)
    
    return result
    
    