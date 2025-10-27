from .. import constants

import numpy as np
pi = np.pi

import copy

import numdifftools as ndt
from scipy.linalg import inv
from scipy.optimize import minimize#, OptimizeResult
from scipy.integrate import quad
from scipy.stats import chi2

from .statistical_measures import minimization_measures
from .fit_initializer import initializer
from .parameters import parameter_set
from .data_prepper import load_dataset, load_barrett_moment
from .pickler import pickle_load_result_dict, pickle_dump_result_dict

from ..dirac_solvers import crosssection_lepton_nucleus_scattering

from .. import nucleus

def fitter(datasets_keys:list,initialization:initializer,barrett_moment_keys=[],monotonous_decrease_precision=np.inf,xi_diff_convergence_limit=1e-4,numdifftools_step=1.e-4,verbose=True,renew=False,cross_section_args={},**minimizer_args):
    ''' **minimzer_args is passed to scipy minimize '''
    # usually: monotonous_decrease_precision=0.04
    # xi_diff_convergence_limit is a deprecated feature and has no effect   
    
    settings_dict = {'datasets':datasets_keys,'datasets_barrett_moment':barrett_moment_keys,'monotonous_decrease_precision':monotonous_decrease_precision,'xi_diff_convergence_limit':xi_diff_convergence_limit,'numdifftools_step':numdifftools_step,**cross_section_args,**minimizer_args}
    
    initial_parameters = parameter_set(initialization.R,initialization.Z,ai=initialization.ai,ai_abs_bound=initialization.ai_abs_bound)
    
    initializer_dict = {'Z':initialization.Z,'A':initialization.A,'R':initialization.R,'N':initialization.N,'xi_ini':initial_parameters.get_xi(),'ai_ini':initial_parameters.get_ai(),'ai_abs_bounds':initial_parameters.ai_abs_bound}
    
    test_dict = {**settings_dict,**initializer_dict}
    visible_keys = ['Z','A','R','N','datasets']
    tracked_keys = list(test_dict.keys())
    
    loaded_results_dict = pickle_load_result_dict(test_dict,tracked_keys,visible_keys)
    
    if (loaded_results_dict is None) or renew:
    
        measures = construct_measures(initialization.Z,initialization.A,datasets_keys,barrett_moment_keys,monotonous_decrease_precision,cross_section_args)
        
        global loss_eval
        loss_eval = 0 
        # define loss function
        def loss_function(xi):
            global loss_eval
            loss_eval+=1
            parameters = parameter_set(initialization.R,initialization.Z,xi=xi,ai_abs_bound=initialization.ai_abs_bound)
            current_nucleus.update_ai(parameters.get_ai())
            loss=0
            for dataset_key in measures:
                loss += measures[dataset_key].loss(current_nucleus)
            if loss_eval%10==0:
                print("Loss (R="+str(current_nucleus.R)+",N="+str(current_nucleus.N_a)+",eval:"+str(loss_eval)+") =",loss)
            return loss

        off_diagonal_covariance=False
        for key in measures:
            off_diagonal_covariance+=measures[key].off_diagonal_covariance

        xi_initial = initial_parameters.get_xi()
        xi_bounds = len(xi_initial)*[(0,1)]
        
        current_nucleus = copy.deepcopy(initialization.nucleus)
        
        converged=False
        rand = 1
        while not converged and rand > 0:
            for key in measures:
                measures[key].set_cov(current_nucleus)    
            print('Starting current fit step (R='+str(current_nucleus.R)+',N='+str(current_nucleus.N_a)+') with loss =',loss_function(xi_initial))
            result = minimize(loss_function,xi_initial,bounds=xi_bounds,**minimizer_args)
            print('Finished current fit step (R='+str(current_nucleus.R)+',N='+str(current_nucleus.N_a)+') with loss =',result.fun)
            
            if not off_diagonal_covariance:
                converged=True
            else:
                xi_diff = result.x - xi_initial
                if np.all(np.abs(xi_diff) < xi_diff_convergence_limit):
                    converged=True
                else:
                    print('Not converged (R='+str(current_nucleus.R)+',N='+str(current_nucleus.N_a)+'): x_f-x_i =',xi_diff)
                    xi_initial = result.x 
            rand = 1.0 + np.random.rand()
        print('Finished fit (R='+str(current_nucleus.R)+',N='+str(current_nucleus.N_a)+'), Calculating Hessian')
        
        Hessian_function = ndt.Hessian(loss_function,step=numdifftools_step)
        hessian = Hessian_function(result.x)
        hessian_inv = inv(hessian)
        covariance_xi = 2*hessian_inv
        
        print('Finished, Constructing results dictionary (R='+str(current_nucleus.R)+',N='+str(current_nucleus.N_a)+')')
        
        out_parameters = parameter_set(initialization.R,initialization.Z,xi=result.x,ai_abs_bound=initialization.ai_abs_bound)
        out_parameters.update_cov_xi_then_cov_ai(covariance_xi)
        out_parameters.set_ai_tilde_from_xi()
        out_parameters.set_ai_from_ai_tilde()
        
        parameters_results={'xi':out_parameters.get_xi(),'ai':out_parameters.get_ai(),'dxi_stat':np.sqrt(out_parameters.cov_xi.diagonal()),'dai_stat':np.sqrt(out_parameters.cov_ai.diagonal()),'cov_xi_stat':out_parameters.cov_xi,'cov_ai_stat':out_parameters.cov_ai}
        
        # calc statistical measures
        chisq, resid, sample_size, dof, redchisq, p_val = {}, {}, {}, {}, {}, {}
        chisq['total'], sample_size['total'] = 0, 0
        for dataset_key in measures:
            resid[dataset_key] = measures[dataset_key].residual(current_nucleus)
            chisq[dataset_key] = measures[dataset_key].loss(current_nucleus)
            sample_size[dataset_key] = len(resid[dataset_key])
            if sample_size[dataset_key] > out_parameters.N_x:
                dof[dataset_key] = sample_size[dataset_key] - out_parameters.N_x
                redchisq[dataset_key] = chisq[dataset_key]/dof[dataset_key]
                p_val[dataset_key] = chi2.sf(chisq[dataset_key],dof[dataset_key])
            chisq['total'] += chisq[dataset_key]
            sample_size['total'] += sample_size[dataset_key]
        dof['total'] = sample_size['total'] - out_parameters.N_x
        redchisq['total'] =  chisq['total']/dof['total']
        p_val['total'] = chi2.sf(chisq['total'],dof['total'])    
        statistics_dict = {'chisq':chisq,'redchisq':redchisq,'p_val':p_val,'dof':dof,'sample_size':sample_size,'resid':resid}
        
        statistics_results={'chisq':chisq['total'],'redchisq':redchisq['total'],'p_val':p_val['total'],'dof':dof['total'],'sample_size':sample_size['total'],'nfev':loss_eval,'statistics_dict':statistics_dict}
        
        values_results={}
        for dataset_key in measures:
            values_results['x_'+dataset_key]=measures[dataset_key].x_data
            values_results['y_'+dataset_key]=measures[dataset_key].test_function_eval(current_nucleus)
        
        # calc radius and barrett moment uncertainties
        r_ch = current_nucleus.charge_radius
        dr_ch = np.sqrt(np.einsum('i,ij,j->',current_nucleus.charge_radius_jacobian,out_parameters.cov_ai,current_nucleus.charge_radius_jacobian))
        
        radius_dict={'r_ch':r_ch,'dr_ch_stat':dr_ch}
        
        barrett_dict={}
        for barrett_moment_key in barrett_moment_keys:
            k, alpha = measures['barrett_moment_'+barrett_moment_key].x_data
            barrett = current_nucleus.barrett_moment(k,alpha)
            barrett_jacob = current_nucleus.barrett_moment_jacobian(k,alpha)           
            dbarrett = np.sqrt(np.einsum('i,ij,j->',barrett_jacob,out_parameters.cov_ai,barrett_jacob))
            barrett_dict={**barrett_dict,'k_'+barrett_moment_key:k,'alpha_'+barrett_moment_key:alpha,'barrett_'+barrett_moment_key:barrett,'dbarrett_'+barrett_moment_key:dbarrett}
            
        results_dict={**settings_dict,**initializer_dict,**statistics_results,**parameters_results,**values_results,**radius_dict,**barrett_dict}
        
        print('Dumping results (R='+str(current_nucleus.R)+',N='+str(current_nucleus.N_a)+')')
        
        pickle_dump_result_dict(results_dict,tracked_keys,visible_keys,overwrite=renew)
    
    else:
        print('Fit with these initial values was already calculated before (R='+str(initialization.R)+',N='+str(initialization.N)+')')
        
        results_dict = loaded_results_dict
    
    return results_dict 

def construct_measures(Z,A,datasets_keys:list,barrett_moment_keys=[],monotonous_decrease_precision=np.inf,cross_section_args={}):
    
    datasets = {}
    barrett_moment_constraint = (len(barrett_moment_keys)>0) # not (barrett_moment_key is None) #
    monotonous_decrease_constraint = (monotonous_decrease_precision<np.inf)
    
    measures={}
    
    for dataset_key in datasets_keys:
        datasets[dataset_key] = {}
        dataset, corr_stat, corr_syst = load_dataset(dataset_key,Z,A,verbose=False)    
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
    
    for dataset_key in datasets:
        measures[dataset_key]=minimization_measures(cross_section,**datasets[dataset_key])
    
    if barrett_moment_constraint:
        barrett_moments = {}
        for barrett_moment_key in barrett_moment_keys:
            barrett_moments['barrett_moment_'+barrett_moment_key]={}
            barrett_dict = load_barrett_moment(barrett_moment_key,Z,A,verbose=False)
            barrett_moments['barrett_moment_'+barrett_moment_key]['x_data'] = (barrett_dict["k"],barrett_dict["alpha"])
            barrett_moments['barrett_moment_'+barrett_moment_key]['y_data'] = barrett_dict["barrett"]
            barrett_moments['barrett_moment_'+barrett_moment_key]['cov_stat_data'] = barrett_dict["dbarrett"]**2
            barrett_moments['barrett_moment_'+barrett_moment_key]['cov_syst_data'] = 0
            def barrett_moment(k_alpha_tuple,nucleus):
                return np.atleast_1d(nucleus.barrett_moment(*k_alpha_tuple))
            measures['barrett_moment_'+barrett_moment_key]=minimization_measures(barrett_moment,**barrett_moments['barrett_moment_'+barrett_moment_key])
        
    if monotonous_decrease_constraint:
        def positive_slope_component_to_radius_squared(_,nucleus):
            integrand = lambda r: -4*pi*r**5/5*nucleus.dcharge_density_dr(r)/nucleus.Z
            integrand_positive_slope = lambda r: np.where(integrand(r)<0,integrand(r),0) 
            positive_slope_component = quad(integrand_positive_slope,0,nucleus.R,limit=1000)[0]
            return np.atleast_1d(positive_slope_component)
        measures['monotonous_decrease']=minimization_measures(positive_slope_component_to_radius_squared,x_data=np.nan,y_data=0,cov_stat_data=monotonous_decrease_precision**2,cov_syst_data=0)    
    
    return measures




def recalc_covariance(fit_result:dict,numdifftools_step=1.e-4,cross_section_args={}):

    datasets_keys, barrett_moment_keys, monotonous_decrease_precision = fit_result['datasets'] , fit_result['datasets_barrett_moment'], fit_result['monotonous_decrease_precision']

    Z, A, R, ai_abs_bounds = fit_result['Z'], fit_result['A'], fit_result['R'], fit_result['ai_abs_bounds']
    ai_bestfit = fit_result['ai']
    xi_bestfit = fit_result['xi']

    measures = construct_measures(Z,A,datasets_keys,barrett_moment_keys,monotonous_decrease_precision,cross_section_args)
    
    def loss_function(xi):
        parameters = parameter_set(R,Z,xi=xi,ai_abs_bound=ai_abs_bounds)
        current_nucleus.update_ai(parameters.get_ai())
        loss=0
        for dataset_key in measures:
            loss += measures[dataset_key].loss(current_nucleus)
        return loss

    current_nucleus = nucleus('FB_stat_calc',Z,A,ai=ai_bestfit,R=R)

    for key in measures:
        measures[key].set_cov(current_nucleus)

    Hessian_function = ndt.Hessian(loss_function,step=numdifftools_step)
    hessian = Hessian_function(xi_bestfit)
    hessian_inv = inv(hessian)
    covariance_xi = 2*hessian_inv

    return covariance_xi

def overwrite_statistical_uncertainties(fit_result, covariance_xi, barrett_moment_keys=[]):
    
    Z, A, R, ai_abs_bounds = fit_result['Z'], fit_result['A'], fit_result['R'], fit_result['ai_abs_bounds']
    ai_fit = fit_result['ai']
    xi_fit = fit_result['xi']
    
    current_nucleus = nucleus('FB_stat_calc',Z,A,ai=ai_fit,R=R)
    
    out_parameters = parameter_set(R,Z,xi=xi_fit,ai_abs_bound=ai_abs_bounds)
    out_parameters.update_cov_xi_then_cov_ai(covariance_xi)
    out_parameters.set_ai_tilde_from_xi()
    out_parameters.set_ai_from_ai_tilde()
    
    parameters_results={'xi':out_parameters.get_xi(),'ai':out_parameters.get_ai(),'dxi_stat':np.sqrt(out_parameters.cov_xi.diagonal()),'dai_stat':np.sqrt(out_parameters.cov_ai.diagonal()),'cov_xi_stat':out_parameters.cov_xi,'cov_ai_stat':out_parameters.cov_ai}
    
    # calc radius and barrett moment uncertainties
    r_ch = current_nucleus.charge_radius
    dr_ch = np.sqrt(np.einsum('i,ij,j->',current_nucleus.charge_radius_jacobian,out_parameters.cov_ai,current_nucleus.charge_radius_jacobian))
    
    radius_dict={'r_ch':r_ch,'dr_ch_stat':dr_ch}
    
    barrett_dict={}
    for barrett_moment_key in barrett_moment_keys:
        k, alpha = fit_result['k_'+barrett_moment_key], fit_result['alpha_'+barrett_moment_key]
        barrett = current_nucleus.barrett_moment(k,alpha)
        barrett_jacob = current_nucleus.barrett_moment_jacobian(k,alpha)           
        dbarrett = np.sqrt(np.einsum('i,ij,j->',barrett_jacob,out_parameters.cov_ai,barrett_jacob))
        barrett_dict={**barrett_dict,'k_'+barrett_moment_key:k,'alpha_'+barrett_moment_key:alpha,'barrett_'+barrett_moment_key:barrett,'dbarrett_'+barrett_moment_key:dbarrett}
    
    fit_result={**fit_result,**parameters_results,**radius_dict,**barrett_dict}
    
    return fit_result    