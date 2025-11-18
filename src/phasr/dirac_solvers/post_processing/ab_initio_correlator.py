from ...config import local_paths

from ... import constants, masses

import numpy as np
pi=np.pi

import os

# the code of this submodule is fairly specialized to the initial specific use case and should be generalized
# no guarantied for the contents and reliability of this submodule
# for a more gernealized version consider the function provided as part of correlation.py

#import numdifftools as ndt
from scipy.interpolate import splev, splrep
from scipy.optimize import minimize
from scipy.linalg import inv

from ...physical_constants.iaea_nds import massofnucleusZN, JPofnucleusZN
from ...nuclei import nucleus
from ...nuclei.parameterizations.numerical import field_ultimate_cutoff#, highenergycutoff_field

from .overlap_integrals import overlap_integral_scalar, overlap_integral_vector, overlap_integral_dipole

from .left_right_asymmetry import left_right_asymmetry_lepton_nucleus_scattering

from functools import partial

def prepare_ab_initio_results(Z,A,folder_path,name=None,r_cut=None,print_radius_check=False): #,r_cut=8
    #
    if name is None:
        name='Z'+str(Z)+'A'+str(A)
    #
    mass_nucleus=massofnucleusZN(Z,A-Z) #in MeV
    spin_nucleus,parity_nucleus=JPofnucleusZN(Z,A-Z)
    #
    # Load files
    AI_files=os.listdir(folder_path)
    AI_models=[]
    for file in AI_files:
        if file.endswith(".csv") and not file.endswith("FF.csv"):
            AI_models.append(file[:-4])
    #
    AI_datasets={}
    for AI_model in AI_models:
        path_par=folder_path+AI_model+'.csv'
        path_FF=folder_path+AI_model+'_FF.csv'
        with open( path_par, "rb" ) as file:
            params_array = np.genfromtxt( file,comments=None,skip_header=0,delimiter=',',names=['par','val'],autostrip=True,dtype=['<U10',float])
        params={param[0]:param[1] for param in params_array}
        with open( path_FF, "rb" ) as file:
            FF0 = np.genfromtxt( file,comments=None,delimiter=',',names=True,autostrip=True,dtype=float)
        AI_datasets[AI_model]={**params,'FF0':FF0}
    #
    #
    print('Loaded datasets:',list(AI_datasets.keys()))
    # norm correction
    for AI_model in AI_datasets:
        FF0=AI_datasets[AI_model]['FF0']
        formfactors=np.copy(FF0)
        for key in FF0.dtype.names:
            try:
                L = int(key[-2])
                if key[1:6] in ['Sigma','Delta']:
                    formfactors[key]*=np.sqrt(4*pi/(2*spin_nucleus+1))
                if L>0: # L=0 are already normalized
                    if key[1:2] in ['M']:
                        formfactors[key]*=1/np.sqrt(2*spin_nucleus+1)
                    if key[1:6] in ['Phipp']:
                        formfactors[key]*=1/np.sqrt(2*spin_nucleus+1)
            except IndexError:
                pass
        AI_datasets[AI_model]['FF']=formfactors
    
    # spline the data
    for AI_model in AI_datasets:
        AI_dict={}
        TCM=AI_datasets[AI_model]['TCM']
        Omega=TCM*4/3
        formfactors=AI_datasets[AI_model]['FF']
        
        multipoles_keys = list(formfactors.dtype.names)
        multipoles_keys.remove('q')
        x_data=formfactors['q']

        # spline and add CMS corrections
        for key in multipoles_keys:
            y_data = formfactors[key]
            y_data_spl = splrep(x_data,y_data,s=0)
            form_factor_spl = partial(CMS_corrected_spline,Omega=Omega,A=A,y_data_spl=y_data_spl)
            form_factor =  partial(field_ultimate_cutoff,R=np.max(x_data),val=0,field_spl=form_factor_spl) # Asymptotic: cutoff to 0
            AI_dict[key] = form_factor
        
        AI_datasets[AI_model]['form_factor_dict']=AI_dict

    # build atoms & calculate densities 
    for AI_model in AI_datasets:

        kws = {} if r_cut is None else {'rrange' : [0.,r_cut,0.05]}
        #kws = {**kws} if q_cut is None else {**kws,'qrange' : [0.,q_cut,1]}
        atom_AI = nucleus(name+"_"+AI_model,Z=Z,A=A,mass=mass_nucleus,spin=spin_nucleus,parity=parity_nucleus,form_factor_dict=AI_datasets[AI_model]['form_factor_dict'],**kws) 
        atom_AI.set_density_dict_from_form_factor_dict()
        atom_AI.set_scalars_from_rho()
        if hasattr(atom_AI,'form_factor') or hasattr(atom_AI,'charge_denstiy'):
            atom_AI.fill_gaps()
        atom_AI.update_dependencies()
        AI_datasets[AI_model]['atom'] = atom_AI 
        
        # identify type
        if 'NIsample' in atom_AI.name:      
            AI_datasets[AI_model]['type'] = 'nonimplausible'
        elif ('EM' in atom_AI.name) or ('sim' in atom_AI.name):
            AI_datasets[AI_model]['type'] = 'magic'
        else:
            AI_datasets[AI_model]['type'] = AI_model
        
        # check radii
        Rn2c = AI_datasets[AI_model]['Rn2c']
        Rp2c = AI_datasets[AI_model]['Rp2c']
        Rso2 = AI_datasets[AI_model]['Rso2']
        Rch2c = r_ch_rpso(Rp2c,Rso2,Z,A)
        
        pres_list=[0]
        if hasattr(atom_AI,'neutron_radius_sq'):
            rn2c = atom_AI.neutron_radius_sq 
            pres_n = np.abs(Rn2c-rn2c)/Rn2c
            pres_list+=[pres_n]
        if hasattr(atom_AI,'proton_radius_sq'):
            rp2c = atom_AI.proton_radius_sq  
            pres_p = np.abs(Rp2c-rp2c)/Rp2c
            pres_list+=[pres_p]
        if hasattr(atom_AI,'charge_radius_sq'):        
            rch2c = atom_AI.charge_radius_sq    
            pres_ch = np.abs(Rch2c-rch2c)/Rch2c
            pres_list+=[pres_ch]
        pres_r2 = np.max(pres_list)
        
        # Warns and lists the radii if the differences are above 1e-3
        if pres_r2>1e-3 or print_radius_check:
            print(('' if print_radius_check else 'Warning: ')+'Some radii ('+AI_model+') are inconsistent at a level of: {:.1e}'.format(pres_r2))
            if hasattr(atom_AI,'neutron_radius_sq'):
                print('rn2  (ref,calc):',Rn2c,rn2c)
            if hasattr(atom_AI,'proton_radius_sq'):
                print('rp2  (ref,calc):',Rp2c,rp2c)
            if hasattr(atom_AI,'charge_radius_sq'):        
                print('rch2 (ref,calc):',Rch2c,rch2c)
        else:
            print('Radii ('+AI_model+') are consistent up to a level of at least: {:.1e}'.format(pres_r2))      
    
    return AI_datasets

def CMS_corrected_spline(q,Omega,A,y_data_spl):
    return splev(q,y_data_spl,ext=0)*F_CMS_Gauss(q,Omega,A)

def translate_old_to_new(AI_datasets,reference_nucleus,q_exp=None,E_exp=None,theta_exp=None,acceptance_exp=None,renew=False,verbose=True,verboseLoad=True,overlap_integral_args={},left_right_asymmetry_args={}):
    
    for AI_model in AI_datasets:
        
        prekeys = list(AI_datasets[AI_model].keys())
        
        if acceptance_exp is None:
            path_correlation_quantities=local_paths.correlation_quantities_paths + "correlation_quantities_"+AI_datasets[AI_model]['atom'].name+'_'+reference_nucleus.name+('_E{:.2f}_theta{:.4f}'.format(E_exp,theta_exp) if (not (E_exp is None) and (not theta_exp is None)) else '')+('_q{:.3f}'.format(q_exp) if q_exp is not None else '')+".txt"
        else:
            path_correlation_quantities=local_paths.correlation_quantities_paths + "correlation_quantities_"+AI_datasets[AI_model]['atom'].name+'_'+reference_nucleus.name+('_E{:.2f}_weighted_mean'.format(E_exp) if (not (E_exp is None)) else '')+('_q{:.3f}'.format(q_exp) if q_exp is not None else '')+".txt"
        
        os.makedirs(os.path.dirname(path_correlation_quantities), exist_ok=True)

        if os.path.exists(path_correlation_quantities) and renew==False:
            with open( path_correlation_quantities, "rb" ) as file:
                correlation_quantities_array = np.genfromtxt( file,comments=None,skip_header=0,delimiter=',',names=['par','val'],autostrip=True,dtype=['<U100',float])
            
            saved_values = {quantity_tuple[0]:quantity_tuple[1] for quantity_tuple in correlation_quantities_array}
            AI_datasets[AI_model]={**AI_datasets[AI_model],**saved_values}
            if verboseLoad:
                print("Loaded (existing) correlation quantities for "+str(AI_model)+" from ",path_correlation_quantities)
            
            saved_keys = list(saved_values.keys())
        else:
            print('path does not exist:',path_correlation_quantities)
            saved_keys = []

        nuc_ref_str = reference_nucleus.name

        for key in saved_keys:
            if key in ['S_p','S_n','V_p','V_n','S_ch','V_ch']:
                new_key = key[:1]+key[2:]+'_rhoch_'+nuc_ref_str
                print(key,'->',new_key)
                AI_datasets[AI_model][new_key] = AI_datasets[AI_model][key]
            if key in ['APV']:
                new_key = 'APV_'+'E{:.2f}_theta{:.4f}'.format(E_exp,theta_exp)+'_rhoch_'+nuc_ref_str
                print(key,'->',new_key)
                AI_datasets[AI_model][new_key] = AI_datasets[AI_model][key]
            if key in ['APV2']:
                new_key = 'APV_'+'E{:.2f}_theta{:.4f}'.format(E_exp,theta_exp)+'_rhoch_'+'from_dataset'
                print(key,'->',new_key)
                AI_datasets[AI_model][new_key] = AI_datasets[AI_model][key]
            if key in ['theta_mean','Qsq_mean','APV_mean']:
                new_key = key[:-4]+'E{:.2f}_weighted_mean'.format(E_exp)+'_rhoch_'+nuc_ref_str
                print(key,'->',new_key)
                AI_datasets[AI_model][new_key] = AI_datasets[AI_model][key]
            if key in ['theta_mean2','Qsq_mean2','APV_mean2']:
                new_key = key[:-5]+'E{:.2f}_weighted_mean'.format(E_exp)+'_rhoch_'+'from_dataset'
                print(key,'->',new_key)
                AI_datasets[AI_model][new_key] = AI_datasets[AI_model][key]

        path_correlation_quantities_new=local_paths.correlation_quantities_paths + "correlation_quantities_"+AI_datasets[AI_model]['atom'].name+".txt"
        
        if renew:
            with open( path_correlation_quantities_new, "w" ) as file:
                file.write('')
        for key in AI_datasets[AI_model]:
            if key not in prekeys:
                if (key not in saved_keys) or renew:
                    with open( path_correlation_quantities_new, "a" ) as file:
                        line='{},{val:.16e}'.format(key,val=AI_datasets[AI_model][key]) #key+','+str(a[key])
                        file.write(line+'\n')
        if verboseLoad:
            print("Correlation quantities (overlap integrals, radii, etc.) saved in ", path_correlation_quantities_new)
    
    



def calculate_correlation_quantities(AI_datasets,reference_nucleus,q_exp=None,E_exp=None,theta_exp=None,acceptance_exp=None,renew=False,verbose=True,verboseLoad=True,overlap_integral_args={},left_right_asymmetry_args={}):
    #
    for AI_model in AI_datasets:
        
        prekeys = list(AI_datasets[AI_model].keys())
        
        if acceptance_exp is None:
            path_correlation_quantities=local_paths.correlation_quantities_paths + "correlation_quantities_"+AI_datasets[AI_model]['atom'].name+'_'+reference_nucleus.name+('_E{:.2f}_theta{:.4f}'.format(E_exp,theta_exp) if (not (E_exp is None) and (not theta_exp is None)) else '')+('_q{:.3f}'.format(q_exp) if q_exp is not None else '')+".txt"
        else:
            path_correlation_quantities=local_paths.correlation_quantities_paths + "correlation_quantities_"+AI_datasets[AI_model]['atom'].name+'_'+reference_nucleus.name+('_E{:.2f}_weighted_mean'.format(E_exp) if (not (E_exp is None)) else '')+('_q{:.3f}'.format(q_exp) if q_exp is not None else '')+".txt"
        
        os.makedirs(os.path.dirname(path_correlation_quantities), exist_ok=True)

        if os.path.exists(path_correlation_quantities) and renew==False:
            with open( path_correlation_quantities, "rb" ) as file:
                correlation_quantities_array = np.genfromtxt( file,comments=None,skip_header=0,delimiter=',',names=['par','val'],autostrip=True,dtype=['<U10',float])
            
            saved_values = {quantity_tuple[0]:quantity_tuple[1] for quantity_tuple in correlation_quantities_array}
            AI_datasets[AI_model]={**AI_datasets[AI_model],**saved_values}
            if verboseLoad:
                print("Loaded (existing) correlation quantities for "+str(AI_model)+" from ",path_correlation_quantities)
            
            saved_keys = list(saved_values.keys())
        else:
            saved_keys = []

        if verbose:
            print('Calculating (additional) correlation quantities for: ',AI_model)
        #
        atom_key = AI_datasets[AI_model]['atom']
        if (not 'rch' in saved_keys) or renew:
            AI_datasets[AI_model]['rch']=atom_key.charge_radius
        if (not 'rchsq' in saved_keys) or renew:
            AI_datasets[AI_model]['rchsq']=atom_key.charge_radius_sq
        if (not 'rp' in saved_keys) or renew:
            AI_datasets[AI_model]['rp']=atom_key.proton_radius
        if (not 'rpsq' in saved_keys) or renew:
            AI_datasets[AI_model]['rpsq']=atom_key.proton_radius_sq
        if (not 'rn' in saved_keys) or renew:
            AI_datasets[AI_model]['rn']=atom_key.neutron_radius
        if (not 'rnsq' in saved_keys) or renew:
            AI_datasets[AI_model]['rnsq']=atom_key.neutron_radius_sq
        if (not 'rw' in saved_keys) or renew:
            AI_datasets[AI_model]['rw']=atom_key.weak_radius
        if (not 'rwsq' in saved_keys) or renew:
            AI_datasets[AI_model]['rwsq']=atom_key.weak_radius_sq
        #
        if q_exp is not None:
            if (not 'Fch_exp' in saved_keys) or renew:
                AI_datasets[AI_model]['Fch_exp']=atom_key.Fch(q_exp,L=0)
            if (not 'Fw_exp' in saved_keys) or renew:
                AI_datasets[AI_model]['Fw_exp']=atom_key.Fw(q_exp,L=0)
        #
        for nuc in ['p','n','ch']:
            #key='M0'+nuc
            if (not 'S_'+nuc in saved_keys) or renew:
                AI_datasets[AI_model]['S_'+nuc] = overlap_integral_scalar(reference_nucleus,nuc,nucleus_response=atom_key,nonzero_electron_mass=True,**overlap_integral_args)
            if (not 'V_'+nuc in saved_keys) or renew:
                AI_datasets[AI_model]['V_'+nuc] = overlap_integral_vector(reference_nucleus,nuc,nucleus_response=atom_key,nonzero_electron_mass=True,**overlap_integral_args)
        #
        if E_exp is not None and theta_exp is not None:
            
            if acceptance_exp is None:
                if (not 'APV' in saved_keys) or renew:
                    AI_datasets[AI_model]['APV'] = left_right_asymmetry_lepton_nucleus_scattering(E_exp,theta_exp,atom_key,reference_nucleus,verbose=True,**left_right_asymmetry_args)
                if (not 'APV2' in saved_keys) or renew:
                    AI_datasets[AI_model]['APV2'] = left_right_asymmetry_lepton_nucleus_scattering(E_exp,theta_exp,atom_key,atom_key,verbose=True,**left_right_asymmetry_args)
            else:
                if (not 'APV_mean' in saved_keys) or renew:
                    AI_datasets[AI_model]['theta_mean'], AI_datasets[AI_model]['Qsq_mean'], AI_datasets[AI_model]['APV_mean'] = left_right_asymmetry_lepton_nucleus_scattering(E_exp,theta_exp,atom_key,reference_nucleus,acceptance=acceptance_exp,verbose=True,**left_right_asymmetry_args)
                if (not 'APV_mean2' in saved_keys) or renew:
                    AI_datasets[AI_model]['theta_mean2'], AI_datasets[AI_model]['Qsq_mean2'], AI_datasets[AI_model]['APV_mean2'] = left_right_asymmetry_lepton_nucleus_scattering(E_exp,theta_exp,atom_key,atom_key,acceptance=acceptance_exp,verbose=True,**left_right_asymmetry_args)
        #
        if renew:
            with open( path_correlation_quantities, "w" ) as file:
                file.write('')
        for key in AI_datasets[AI_model]:
            if key not in prekeys:
                if (key not in saved_keys) or renew:
                    with open( path_correlation_quantities, "a" ) as file:
                        line='{},{val:.16e}'.format(key,val=AI_datasets[AI_model][key]) #key+','+str(a[key])
                        file.write(line+'\n')
        if verboseLoad:
            print("Correlation quantities (overlap integrals, radii, etc.) saved in ", path_correlation_quantities)
    
    return AI_datasets

def r_ch_rpso(r2p,r2so,Z,A):
    return r2p + constants.rsq_p + ((A-Z)/Z)*constants.rsq_n + 3*constants.hc**2/(4*masses.mN**2) + r2so

def b_cm(Omega,A,mN=938.9):#MeV
    return np.sqrt(1/(A*mN*Omega))

def F_CMS_Gauss(q,Omega,A):
    b=b_cm(Omega,A)
    return np.exp((b*q/2)**2)

def linear_model(x,m,b):
    return x * m + b

def fit_linear_correlation(arr_dict,x_str,y_str,x_offset,**minimizer_args): #,numdifftools_step=1.e-4,scale_yerr=True
    
    x_data_unnormalized=arr_dict[x_str] + x_offset
    y_data_unnormalized=arr_dict[y_str]
    
    # normalize data in x any y direction 
    x_data_mean = np.mean(x_data_unnormalized)
    x_data_std = np.std(x_data_unnormalized)
    y_data_mean = np.mean(y_data_unnormalized)
    y_data_std = np.std(y_data_unnormalized)
    x_data = (x_data_unnormalized - x_data_mean) / x_data_std
    y_data = (y_data_unnormalized - y_data_mean) / y_data_std
    
    # fit normalized 
    y_error=np.std(y_data)
    m_ini = -np.std(y_data)/np.std(x_data) # somewhat arbitrary
    b_ini = np.mean(y_data) # somewhat arbitrary
    xi_initial = np.array([m_ini,b_ini])
    
    def residuals(xi):
        model = linear_model(x_data,xi[0],xi[1])
        residuals = (model - y_data)/y_error
        return residuals
    
    def loss(xi):
        return np.sum(residuals(xi)**2)
    
    result = minimize(loss,xi_initial,**minimizer_args) 
    
    # if fit statistics become relevant in the future reactivate 
    # Hessian_function = ndt.Hessian(loss,step=numdifftools_step)
    # hessian = Hessian_function(result.x)
    # hessian_inv = inv(hessian)
    # covariance_xi = 2*hessian_inv
    # unused, what is the correct way to normalize this? 
    # covar = covariance_xi * y_error**2 * (redchisq if scale_yerr else 1.)
    # ,'cov':covar
    # chisq = loss(xi)
    # dof = len(resid) - len(xi)
    # redchisq = chisq / dof
    # ,'residual':resid,'redchisq':redchisq
    
    xi = result.x
    m_normalized, b_normalized = xi[0], xi[1]
    
    # change back to unnormalized coordinates
    m = m_normalized * (y_data_std/x_data_std)
    b = b_normalized * y_data_std + y_data_mean - m_normalized * (y_data_std/x_data_std) * x_data_mean
    
    resid = residuals(xi)
    db = np.std(resid) * y_error * y_data_std
    
    results={'val':b,'dval':db,'m':m,'x_str':x_str,'y_str':x_str}
    return results

def AbInitioCorrelator(AI_datasets,x_str='rchsq',x_offset=0,y_strs=None,corr_skin=False,**minimizer_args): #,return_all=False
    #
    val_arr={}
    #
    val_arr[x_str]=np.array([])
    for AI_model in AI_datasets:
        
        if x_str in ['rn-rp','rw-rch']:
            rsqi=AI_datasets[AI_model][x_str[:2]]-AI_datasets[AI_model][x_str[3:]]
            val_arr[x_str]=np.append(val_arr[x_str],rsqi)
        else:
            rsqi=AI_datasets[AI_model][x_str]
            val_arr[x_str]=np.append(val_arr[x_str],rsqi)
    
    results_dict={}
    
    if y_strs is None:
        if not corr_skin:
            for ov in ['S','V']:
                for nuc in ['p','n']:
                    val_arr[ov+nuc]=np.array([])
                    key = ov+'_'+nuc
                    for AI_model in AI_datasets:
                        ovi=AI_datasets[AI_model][key]
                        val_arr[ov+nuc]=np.append(val_arr[ov+nuc],ovi)
                    results_dict[ov+nuc] = fit_linear_correlation(val_arr,x_str,ov+nuc,x_offset,**minimizer_args)
            for r2 in ['rpsq','rnsq','rwsq']: 
                key = r2
                val_arr[key]=np.array([])
                for AI_model in AI_datasets:
                    r2i=AI_datasets[AI_model][key]
                    val_arr[key]=np.append(val_arr[key],r2i)
                results_dict[key] = fit_linear_correlation(val_arr,x_str,key,x_offset,**minimizer_args) 
        else:
            for rdiff in ['rn-rp','rw-rch']:
                key = rdiff
                key1 = key[:2]
                key2 = key[3:]
                val_arr[key]=np.array([])
                for AI_model in AI_datasets:
                    rdiffi=AI_datasets[AI_model][key1]-AI_datasets[AI_model][key2]
                    val_arr[key]=np.append(val_arr[key],rdiffi)
                results_dict[key] = fit_linear_correlation(val_arr,x_str,key,x_offset,**minimizer_args)   
    else:
        for y_str in y_strs:
            key = y_str
            val_arr[key]=np.array([])
            for AI_model in AI_datasets:
                yi=AI_datasets[AI_model][key]
                val_arr[key]=np.append(val_arr[key],yi)
            results_dict[key] = fit_linear_correlation(val_arr,x_str,key,x_offset,**minimizer_args)
            
    return results_dict, val_arr