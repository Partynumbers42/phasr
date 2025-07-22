from ..config import local_paths

from .. import constants

import numpy as np
pi = np.pi

import copy


import numdifftools as ndt
from scipy.linalg import inv

from scipy.optimize import minimize, OptimizeResult
from scipy.integrate import quad

from .statistical_measures import minimization_measures
from .initializer import initializer
from .parameters import parameter_set
from .data_prepper import load_dataset, load_barrett_moment

from ..dirac_solvers import crosssection_lepton_nucleus_scattering

# TODO rigorous check what nucleus is used at which step of the fit iteration

def fitter(datasets_keys:list,initialization:initializer,barrett_moment_key=None,monotonous_decrease_precision=np.inf,xi_diff_convergence_limit=1e-4,numdifftools_step=1.e-4,verbose=True,cross_section_args={},**minimizer_args):
    ''' **args is passed to scipy minimize '''
    
    # default: monotonous_decrease_precision=0.04
    
    initial_parameters = parameter_set(initialization.R,initialization.Z,ai=initialization.ai,ai_abs_bound=initialization.ai_abs_bound)
    current_nucleus = copy.deepcopy(initialization.nucleus) 
    
    datasets = {}
    barrett_moment_constraint = not (barrett_moment_key is None) #(len(barrett_moment_keys)>0)
    monotonous_decrease_constraint = (monotonous_decrease_precision<np.inf)

    for dataset_key in datasets_keys:
        datasets[dataset_key] = {}
        dataset, corr_stat, corr_syst = load_dataset(dataset_key,initialization.Z,initialization.A,verbose=False)    
        dy_stat, dy_syst = dataset[:,3], dataset[:,4]
        datasets[dataset_key]['x_data'] = dataset[:,(0,1)]
        datasets[dataset_key]['y_data'] = dataset[:,2]
        datasets[dataset_key]['cov_stat_data'] = np.einsum('i,ij,j->ij',dy_stat,corr_stat,dy_stat)
        datasets[dataset_key]['cov_syst_data'] = np.einsum('i,ij,j->ij',dy_syst,corr_syst,dy_syst)
    
    def cross_section(energy_and_theta,nucleus):
        energies = energy_and_theta[:,0]
        thetas = energy_and_theta[:,1]
        cross_section = np.zeros(len(energies))
        for energy in np.unique(energies):
            mask = (energy==energies)
            cross_section[mask] = crosssection_lepton_nucleus_scattering(energy,thetas[mask],nucleus,**cross_section_args)*constants.hc**2
        return cross_section
    
    measures={}
    for dataset_key in datasets:
        measures[dataset_key]=minimization_measures(cross_section,**datasets[dataset_key])
    
    if barrett_moment_constraint:
        barrett_moments = {}
        #for barrett_moment_key in barrett_moment_keys:#current implementation of nucleus does only allow for one value for k, alpha (change?)
        barrett_moments['barrett_moment']={}
        barrett_dict = load_barrett_moment(barrett_moment_key,initialization.Z,initialization.A,verbose=False)
        barrett_moments['barrett_moment']['x_data'] = np.nan
        barrett_moments['barrett_moment']['y_data'] = barrett_dict["barrett"]
        barrett_moments['barrett_moment']['cov_stat_data'] = barrett_dict["dbarrett"]**2
        barrett_moments['barrett_moment']['cov_syst_data'] = 0
        current_nucleus.update_k_and_alpha_barrett(barrett_dict["k"],barrett_dict["alpha"])
        def barrett_moment(_,nucleus):
            return np.atleast_1d(nucleus.barrett_moment)
        measures['barrett_moment']=minimization_measures(barrett_moment,**barrett_moments['barrett_moment'])
    
    if monotonous_decrease_constraint:
        def positive_slope_component_to_radius_squared(_,nucleus):
            integrand = lambda r: -4*pi*r**5/5*nucleus.dcharge_density_dr(r)/nucleus.Z
            integrand_positive_slope = lambda r: np.where(integrand(r)<0,integrand(r),0) 
            positive_slope_component = quad(integrand_positive_slope,0,nucleus.R,limit=1000)[0]
            return np.atleast_1d(positive_slope_component)
        measures['monotonous_decrease']=minimization_measures(positive_slope_component_to_radius_squared,x_data=np.nan,y_data=0,cov_stat_data=monotonous_decrease_precision**2,cov_syst_data=0)
        
    def loss_function(xi):
        
        parameters = parameter_set(initialization.R,initialization.Z,xi=xi,ai_abs_bound=initialization.ai_abs_bound)
        current_nucleus.update_ai(parameters.get_ai())
        
        loss=0
        for dataset_key in measures:
            loss += measures[dataset_key].loss(current_nucleus)

        #print("loss",loss)
        
        return loss

    xi_initial = initial_parameters.get_xi()
    xi_bounds = len(xi_initial)*[(0,1)]
    
    off_diagonal_covariance=False
    for key in measures:
        off_diagonal_covariance+=measures[key].off_diagonal_covariance
    
    if verbose:
        for key in measures:
            measures[key].set_cov(current_nucleus)
        print('Starting fit with initial loss =',loss_function(xi_initial))
    
    converged=False
    while not converged:
        
        for key in measures:
            measures[key].set_cov(current_nucleus)
            
        result = minimize(loss_function,xi_initial,bounds=xi_bounds,**minimizer_args)
        
        print('Finished current fit step with loss =',result.fun)
        
        if not off_diagonal_covariance:
            converged=True
        else:
            xi_diff = result.x - xi_initial
            if np.all(np.abs(xi_diff) < xi_diff_convergence_limit):
                converged=True
            else:
                print('Not converged: x_f-x_i =',xi_diff)
                xi_initial = result.x
                
    
    print('Finished fit, Calculating hessian')
    
    Hessian_function = ndt.Hessian(loss_function,step=numdifftools_step)
    hessian = Hessian_function(result.x)
    hessian_inv = inv(hessian)
    covariance = 2*hessian_inv
    
    print('Finished, Returning results')
    
    out_parameters = parameter_set(initialization.R,initialization.Z,xi=result.x,ai_abs_bound=initialization.ai_abs_bound)
    
    out_parameters.set_cov_xi(covariance)
    out_parameters.set_ai_tilde_from_xi()
    out_parameters.set_ai_from_ai_tilde()
    
    return result, current_nucleus, measures, out_parameters, covariance
    
    