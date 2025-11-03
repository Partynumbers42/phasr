import numpy as np
pi = np.pi

from .statistical_measures import minimization_measures
from .parameters import parameter_set
from .. import nucleus

import numdifftools as ndt
from scipy.linalg import inv

from scipy.optimize import minimize


def generate_systematic_errorband_exact(best_key:str,best_results:dict,rrange=[0,15,1e-1],select='all'):
    
    best_result = best_results[best_key]
    nuc_best_result = nucleus('temp_nuc_best',Z=best_result['Z'],A=best_result['A'],ai=best_result['ai'],R=best_result['R'])
    
    r = np.arange(*rrange)
    rho_best = nuc_best_result.charge_density(r)

    band_syst_max=-np.inf*np.ones(len(r))
    band_syst_min=np.inf*np.ones(len(r))

    for key in best_results:
        
        result = best_results[key]
        nuc_result = nucleus('temp_nuc_'+key,Z=result['Z'],A=result['A'],ai=result['ai'],R=result['R'])
        rho=nuc_result.charge_density(r)
        band_syst_max=np.maximum(band_syst_max,rho)
        band_syst_min=np.minimum(band_syst_min,rho)

    drho_best_syst_upper=(band_syst_max-rho_best)
    drho_best_syst_lower=(rho_best-band_syst_min)
    drho_best_syst=np.max([drho_best_syst_upper,drho_best_syst_lower],axis=0)
    
    if select == 'all':
        return drho_best_syst_upper, drho_best_syst_lower, drho_best_syst
    elif select == 'upper':
        return drho_best_syst_upper
    elif select == 'lower':
        return drho_best_syst_lower
    elif select == 'max':
        return drho_best_syst
    else:
        raise ValueError('Unknown keyword')

def uncertainty_band(r,cov_ai,nucleus):
    jacobian = nucleus.charge_density_jacobian(r)
    drho =  np.sqrt(np.einsum('ij,ik,kj->j',jacobian,cov_ai,jacobian))
    return np.where(drho==drho, drho, 0)

def fit_systematic_errorband(best_key:str,best_results:dict,Omega=10,rrange=[0,15,1e-1],numdifftools_step=1.e-4):

    r=np.arange(*rrange)
    
    drho_syst_exact={}
    drho_syst_exact['upper'], drho_syst_exact['lower'], drho_syst_exact[''] = generate_systematic_errorband_exact(best_key,best_results,rrange)

    syst_measures = {}
    exact_uncertainty_band = {}
    for syst_key in ['upper','lower','']:
        exact_uncertainty_band[syst_key] = {}
        exact_uncertainty_band[syst_key]['x_data'] = r
        exact_uncertainty_band[syst_key]['y_data'] = drho_syst_exact[syst_key]
        exact_uncertainty_band[syst_key]['cov_stat_data'] = 0
        exact_uncertainty_band[syst_key]['cov_syst_data'] = 0
        syst_measures[syst_key]=minimization_measures(uncertainty_band,**exact_uncertainty_band[syst_key])
    
    best_result = best_results[best_key]
    nuc_best_result = nucleus('temp_nuc',Z=best_result['Z'],A=best_result['A'],ai=best_result['ai'],R=best_result['R'])

    cov_xi_best = best_result['cov_xi_stat']
    dxi_best = best_result['dxi_stat']
    corr_xi = np.einsum('i,ij,j->ij',1/dxi_best,cov_xi_best,1/dxi_best)

    parameters = parameter_set(best_result['R'],best_result['Z'],xi=best_result['xi'],ai_abs_bound=best_result['ai_abs_bounds'])        

    dxi_initial = dxi_best
    dxi_bounds = len(dxi_initial)*[(0,None)]
    
    out_dict = {}

    for syst_key in ['upper','lower','']:
        def loss_function(dxi):
            cov_xi =  np.einsum('i,ij,j->ij',dxi,corr_xi,dxi)
            parameters.update_cov_xi_then_cov_ai(cov_xi)
            residual = syst_measures[syst_key].residual(parameters.get_cov_ai(),nuc_best_result,weighted=False)
            return np.sum(residual**2 * (1 + np.where(residual < 0,Omega**2,0) ) )
        
        result = minimize(loss_function,dxi_initial,bounds=dxi_bounds)

        # currently not done b/c uncertanty estimation does not work properly
        #Hessian_function = ndt.Hessian(loss_function,step=numdifftools_step)
        #hessian = Hessian_function(result.x)
        #hessian_inv = inv(hessian)
        #cov_dxi = 2*hessian_inv
        #ddxi = np.sqrt(cov_dxi.diagonal()) 
        #dxi_fit = round_positive_by_error(result.x,ddxi,1)
        
        dxi_fit = result.x
        cov_xi_fit =  np.einsum('i,ij,j->ij',dxi_fit,corr_xi,dxi_fit)
        
        parameters.update_cov_xi_then_cov_ai(cov_xi_fit)    
        cov_ai_fit = parameters.get_cov_ai()
        dai_fit = np.sqrt(cov_ai_fit.diagonal()) 
        
        out_dict['dxi_syst'+('_' if len(syst_key)>0 else '')+syst_key] = dxi_fit
        out_dict['dai_syst'+('_' if len(syst_key)>0 else '')+syst_key] = dai_fit
        out_dict['cov_xi_syst'+('_' if len(syst_key)>0 else '')+syst_key] = cov_xi_fit
        out_dict['cov_ai_syst'+('_' if len(syst_key)>0 else '')+syst_key] = cov_ai_fit

    return out_dict

# currently unused
def round_positive_by_error(val,err,offset=1):
    decimals=-np.log10(err)+offset
    return np.true_divide(np.rint(val * 10.**decimals.astype(int)), 10.**decimals.astype(int))

def add_systematic_uncertainties(best_key:str,best_results:dict,rbin=1e-1,**args):

    best_result = best_results[best_key]
    
    cov_xi_stat = best_result['cov_xi_stat']
    dxi_stat = best_result['dxi_stat']
    corr_xi = np.einsum('i,ij,j->ij',1/dxi_stat,cov_xi_stat,1/dxi_stat)
    redchisq_fit = best_result['redchisq']

    parameters = parameter_set(best_result['R'],best_result['Z'],xi=best_result['xi'],ai_abs_bound=best_result['ai_abs_bounds'])        
    
    rrange = [0,best_result['R'],rbin] 
    syst_dict = fit_systematic_errorband(best_key,best_results,rrange=rrange,**args)
    best_result = {**best_result, **syst_dict}

    # combine uncertainties to model uncertainties
    for syst_key in ['','_upper','_lower']:
        dxi_syst = best_result['dxi_syst'+syst_key]
        dxi_model = np.sqrt(redchisq_fit*dxi_stat**2+dxi_syst**2)
        cov_xi_model=np.einsum('i,ij,j->ij',dxi_model,corr_xi,dxi_model)
        
        parameters.update_cov_xi_then_cov_ai(cov_xi_model)    
        cov_ai_model = parameters.get_cov_ai()
        dai_model = np.sqrt(cov_ai_model.diagonal()) 
        
        best_result['dxi_model'+syst_key] = dxi_model
        best_result['dai_model'+syst_key] = dai_model
        best_result['cov_xi_model'+syst_key] = cov_xi_model
        best_result['cov_ai_model'+syst_key] = cov_ai_model
    
    nuc_best_result = nucleus('temp_nuc_best',Z=best_result['Z'],A=best_result['A'],ai=best_result['ai'],R=best_result['R'])
    
    # propagate uncertainties to charge radius
    r_ch_jacobian = nuc_best_result.charge_radius_jacobian

    for syst_key in ['','_upper','_lower']:
        
        cov_ai_syst = best_result['cov_ai_syst'+syst_key]
        dr_ch_syst =  np.sqrt(np.einsum('i,ik,k->',r_ch_jacobian,cov_ai_syst,r_ch_jacobian))
        best_result['dr_ch_syst'+syst_key] = dr_ch_syst
        
        cov_ai_model = best_result['cov_ai_model'+syst_key]
        dr_ch_model =  np.sqrt(np.einsum('i,ik,k->',r_ch_jacobian,cov_ai_model,r_ch_jacobian))
        best_result['dr_ch_model'+syst_key] = dr_ch_model
    
    # propagate uncertainties to barrett moment
    barrett_tuples = [(k_key[2:],best_result[k_key],best_result[alpha_key]) for k_key in best_result.keys() if k_key.startswith('k_') for alpha_key in best_result.keys() if alpha_key.startswith('alpha_') and alpha_key.endswith(k_key[2:])]

    for (barrett_moment_key,k_barrett,alpha_barrett) in barrett_tuples:
        
        barrett_jacobian = nuc_best_result.barrett_moment_jacobian(k_barrett,alpha_barrett)
        
        for syst_key in ['','_upper','_lower']:
        
            cov_ai_syst = best_result['cov_ai_syst'+syst_key]
            dbarrett_syst =  np.sqrt(np.einsum('i,ik,k->',barrett_jacobian,cov_ai_syst,barrett_jacobian))
            best_result['dbarrett_'+barrett_moment_key+'_syst'+syst_key] = dbarrett_syst
            
            cov_ai_model = best_result['cov_ai_model'+syst_key]
            dbarrett_model =  np.sqrt(np.einsum('i,ik,k->',barrett_jacobian,cov_ai_model,barrett_jacobian))
            best_result['dbarrett_'+barrett_moment_key+'_model'+syst_key] = dbarrett_model
            
    # add systematics from the extremes of all fits and their distance to the best fit 
    
    # for charge radius
    r_ch_best = best_result['r_ch']
    r_ch_min = r_ch_best
    r_ch_max = r_ch_best

    for key in best_results:
        r_ch_i = best_results[key]['r_ch']
        if r_ch_i>r_ch_max:
            r_ch_max=r_ch_i
        elif r_ch_i<r_ch_min:
            r_ch_min=r_ch_i
    
    dr_ch_dist_upper = r_ch_max-r_ch_best
    dr_ch_dist_lower = r_ch_best-r_ch_min
    dr_ch_dist = np.max([dr_ch_dist_upper,dr_ch_dist_lower],axis=0)

    best_result['dr_ch_dist']=dr_ch_dist
    best_result['dr_ch_dist_upper']=dr_ch_dist_upper
    best_result['dr_ch_dist_lower']=dr_ch_dist_lower

    # for barrett moments
    barrett_keys = [k_key[2:] for k_key in best_result.keys() if k_key.startswith('k_') for alpha_key in best_result.keys() if alpha_key.startswith('alpha_') and alpha_key.endswith(k_key[2:])]
    for barrett_moment_key in barrett_keys:
        barrett_moment_best = best_result['barrett_'+barrett_moment_key]
        barrett_moment_min = barrett_moment_best
        barrett_moment_max = barrett_moment_best

        for key in best_results:
            barrett_moment_i = best_results[key]['barrett_'+barrett_moment_key]
            if barrett_moment_i>barrett_moment_max:
                barrett_moment_max=barrett_moment_i
            elif barrett_moment_i<barrett_moment_min:
                barrett_moment_min=barrett_moment_i

        dbarrett_moment_dist_upper = barrett_moment_max-barrett_moment_best
        dbarrett_moment_dist_lower = barrett_moment_best-barrett_moment_min
        dbarrett_moment_dist = np.max([dbarrett_moment_dist_upper,dbarrett_moment_dist_lower],axis=0)

        best_result['dbarrett_'+barrett_moment_key+'_dist']=dbarrett_moment_dist
        best_result['dbarrett_'+barrett_moment_key+'_dist_upper']=dbarrett_moment_dist_upper
        best_result['dbarrett_'+barrett_moment_key+'_dist_lower']=dbarrett_moment_dist_lower

    return best_result
    

    