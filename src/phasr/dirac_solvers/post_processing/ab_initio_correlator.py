from ...config import local_paths

from ... import constants, masses

import numpy as np
pi=np.pi

import os

from scipy.interpolate import splev, splrep

from ...physical_constants.iaea_nds import massofnucleusZN, JPofnucleusZN
from ...nuclei import nucleus
from ...nuclei.parameterizations.numerical import field_ultimate_cutoff#, highenergycutoff_field

from .overlap_integrals import overlap_integral_scalar, overlap_integral_vector, overlap_integral_dipole

from .left_right_asymmetry import left_right_asymmetry_lepton_nucleus_scattering

from functools import partial

def prepare_ab_initio_results(Z,A,folder_path,name=None,r_cut=None): #,r_cut=8
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
                if L>0: # L=0 are already normalised
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
            #def form_factor_spl(q,y_data_spl=y_data_spl,Omega=Omega): return splev(q,y_data_spl,ext=0)*F_CMS_Gauss(q,Omega,A)
            form_factor_spl = partial(CMS_corrected_spline,Omega=Omega,A=A,y_data_spl=y_data_spl)
            form_factor =  partial(field_ultimate_cutoff,R=np.max(x_data),val=0,field_spl=form_factor_spl) # Asymptotic: cutoff to 0
            #form_factor =  highenergycutoff_field(form_factor_spl,R=np.max(x_data),val=0)
            AI_dict[key] = form_factor
        
        AI_datasets[AI_model]['form_factor_dict']=AI_dict

    # build atoms & calculate densities 
    for AI_model in AI_datasets:

        kws = {} if r_cut is None else {'rrange' : [0.,r_cut,0.05]}
        atom_AI = nucleus(name+"_"+AI_model,Z=Z,A=A,mass=mass_nucleus,spin=spin_nucleus,parity=parity_nucleus,form_factor_dict=AI_datasets[AI_model]['form_factor_dict'],**kws) 
        atom_AI.set_density_dict_from_form_factor_dict()
        atom_AI.fill_gaps()
        AI_datasets[AI_model]['atom'] = atom_AI 
        
        # identify type
        if 'NIsample' in atom_AI.name:      
            AI_datasets[AI_model]['type'] = 'nonimplausible'
        elif ('EM' in atom_AI.name) or ('sim' in atom_AI.name):
            AI_datasets[AI_model]['type'] = 'magic'
        else:
            AI_datasets[AI_model]['type'] = AI_model
        # more than 3 groups ?
        
        # check radii
        Rn2c = AI_datasets[AI_model]['Rn2c']
        Rp2c = AI_datasets[AI_model]['Rp2c']
        Rso2 = AI_datasets[AI_model]['Rso2']
        Rch2c = r_ch_rpso(Rp2c,Rso2,Z,A)
        
        rn2c = atom_AI.neutron_radius_sq 
        rp2c = atom_AI.proton_radius_sq  
        rch2c = atom_AI.charge_radius_sq
        
        pres_n = np.abs(Rn2c-rn2c)/Rn2c
        pres_p = np.abs(Rp2c-rp2c)/Rp2c
        pres_ch = np.abs(Rch2c-rch2c)/Rch2c
        pres_r2 = np.max([pres_n,pres_p,pres_ch])
        
        # Warns and lists the radii if the differences are above 1e-3
        if pres_r2>1e-3:
            print('Warning: Some radii ('+AI_model+') are inconsistent at a level of: {:.1e}'.format(pres_r2))
            print('rn2  (ref,calc):',Rn2c,rn2c)
            print('rp2  (ref,calc):',Rp2c,rp2c)
            print('rch2 (ref,calc):',Rch2c,rch2c)
        else:
            print('Radii ('+AI_model+') are consistent up to a level of at least: {:.1e}'.format(pres_r2))      
    
    return AI_datasets

def CMS_corrected_spline(q,Omega,A,y_data_spl):
    return splev(q,y_data_spl,ext=0)*F_CMS_Gauss(q,Omega,A)

def calculate_correlation_quantities(AI_datasets,reference_nucleus,q_exp=None,E_exp=None,theta_exp=None,renew=False,verbose=True,verboseLoad=True,left_right_asymmetry_args={}):
    #
    for AI_model in AI_datasets:
        
        prekeys = list(AI_datasets[AI_model].keys())
        path_correlation_quantities=local_paths.correlation_quantities_paths + "correlation_quantities_"+AI_datasets[AI_model]['atom'].name+'_'+reference_nucleus.name+('_E{:.2f}_theta{:.4f}'.format(E_exp,theta_exp) if (not (E_exp is None) and (not theta_exp is None)) else '')+('_q{:.3f}'.format(q_exp) if q_exp is not None else '')+".txt"
        
        os.makedirs(os.path.dirname(path_correlation_quantities), exist_ok=True)

        if os.path.exists(path_correlation_quantities) and renew==False:
            with open( path_correlation_quantities, "rb" ) as file:
                correlation_quantities_array = np.genfromtxt( file,comments=None,skip_header=0,delimiter=',',names=['par','val'],autostrip=True,dtype=['<U10',float])
            
            AI_datasets[AI_model]={**AI_datasets[AI_model],**{quantity_tuple[0]:quantity_tuple[1] for quantity_tuple in correlation_quantities_array}}
            if verboseLoad:
                print("Loaded correlation quantities for "+str(AI_model)+" from ",path_correlation_quantities)
        else:
            if verbose:
                print('Calculating correlation quantities for: ',AI_model)
            #
            atom_key = AI_datasets[AI_model]['atom']
            if not 'rch' in prekeys:
                AI_datasets[AI_model]['rch']=atom_key.charge_radius
            if not 'rchsq' in prekeys:
                AI_datasets[AI_model]['rchsq']=atom_key.charge_radius_sq
            if not 'rp' in prekeys:
                AI_datasets[AI_model]['rp']=atom_key.proton_radius
            if not 'rpsq' in prekeys:
                AI_datasets[AI_model]['rpsq']=atom_key.proton_radius_sq
            if not 'rn' in prekeys:
                AI_datasets[AI_model]['rn']=atom_key.neutron_radius
            if not 'rnsq' in prekeys:
                AI_datasets[AI_model]['rnsq']=atom_key.neutron_radius_sq
            if not 'rw' in prekeys:
                AI_datasets[AI_model]['rw']=atom_key.weak_radius
            if not 'rwsq' in prekeys:
                AI_datasets[AI_model]['rwsq']=atom_key.weak_radius_sq
            #
            if q_exp is not None:
                if not 'Fch_exp' in prekeys:
                    AI_datasets[AI_model]['Fch_exp']=atom_key.Fch(q_exp,L=0)
                if not 'Fw_exp' in prekeys:
                    AI_datasets[AI_model]['Fw_exp']=atom_key.Fw(q_exp,L=0)
            #
            for nuc in ['p','n','ch']:
                #key='M0'+nuc
                if not 'S_'+nuc in prekeys:
                    AI_datasets[AI_model]['S_'+nuc] = overlap_integral_scalar(reference_nucleus,nuc,nucleus_response=atom_key,nonzero_electron_mass=True)
                if not 'V_'+nuc in prekeys:
                    AI_datasets[AI_model]['V_'+nuc] = overlap_integral_vector(reference_nucleus,nuc,nucleus_response=atom_key,nonzero_electron_mass=True)
            #
            if E_exp is not None and theta_exp is not None:
                if not 'APV' in prekeys:
                    AI_datasets[AI_model]['APV'] = left_right_asymmetry_lepton_nucleus_scattering(E_exp,theta_exp,atom_key,reference_nucleus,verbose=True,**left_right_asymmetry_args)
                if not 'APV2' in prekeys:
                    AI_datasets[AI_model]['APV2'] = left_right_asymmetry_lepton_nucleus_scattering(E_exp,theta_exp,atom_key,atom_key,verbose=True,**left_right_asymmetry_args)
            #
            if renew:
                with open( path_correlation_quantities, "w" ) as file:
                    file.write('')
            for key in AI_datasets[AI_model]:
                if key not in prekeys:
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


def linear_model(x,paramdict):
    
    m=paramdict['m']
    b=paramdict['b']
    
    return m * x + b

def residual(params, x, data, yerr):
    model = linear_model(x,params)
    
    return (model-data)/yerr

from lmfit import minimize, Parameters

def AbInitioCorrelator(AI_datasets,x_str='rchsq',x_offset=0,y_strs=None,scale_yerr=True,corr_skin=False): #,return_all=False
    #
    scale_yerr=True    
    #
    ov_arr={}
    #
    ov_arr[x_str]=np.array([])
    for AI_model in AI_datasets:
        rsqi=AI_datasets[AI_model][x_str]
        ov_arr[x_str]=np.append(ov_arr[x_str],rsqi)
    
    
    results_dict={}
    
    if y_strs is None:
        
        if not corr_skin:
        
            for ov in ['S','V']:
                for nuc in ['p','n']:

                    ov_arr[ov+nuc]=np.array([])
                    key = ov+'_'+nuc
                    for AI_model in AI_datasets:
                        ovi=AI_datasets[AI_model][key]
                        ov_arr[ov+nuc]=np.append(ov_arr[ov+nuc],ovi)
                    #
                    x=ov_arr[x_str] + x_offset
                    data=ov_arr[ov+nuc]
                    #
                    params = Parameters()
                    
                    params.add('m', value=-np.std(data)/np.std(x))
                    params.add('b', value=np.mean(data))
                    #
                    yerr=0*x+1.0 # important that this is 1 such that the residuals are unnormalised
                    out = minimize(residual, params, args=(x, data, yerr),scale_covar=scale_yerr)
                    
                    m=out.params['m'].value
                    b=out.params['b'].value
                    resid=out.residual
                    redchi=out.redchi
                    db=np.std(resid)
                    covar=out.covar
                    
                    results={'I':b,'dI':db,'residual':resid,'redchisq':redchi,'m':m,'covar':covar,'x_str':x_str}
                    results_dict[ov+nuc] = results
            
            for r2 in ['rpsq','rnsq','rwsq','APV','APV2']:
                key = r2
                ov_arr[key]=np.array([])
                for AI_model in AI_datasets:
                    r2i=AI_datasets[AI_model][key]
                    ov_arr[key]=np.append(ov_arr[key],r2i)
                #
                x=ov_arr[x_str] + x_offset
                data=ov_arr[key]
                #
                params = Parameters()
                
                params.add('m', value=-np.std(data)/np.std(x))
                params.add('b', value=np.mean(data))
                #
                yerr=0*x+1.0 # important that this is 1 such that the residuals are unnormalised
                out = minimize(residual, params, args=(x, data, yerr),scale_covar=scale_yerr)
                
                m=out.params['m'].value
                b=out.params['b'].value
                resid=out.residual
                redchi=out.redchi
                db=np.std(resid)
                covar=out.covar
                
                results={'I':b,'dI':db,'residual':resid,'redchisq':redchi,'m':m,'covar':covar,'x_str':x_str}
                results_dict[key] = results
    
        else:
        
            for rdiff in ['rn-rp','rw-rch']:
                key = rdiff
                key1 = key[:2]
                key2 = key[3:]
                ov_arr[key]=np.array([])
                for AI_model in AI_datasets:
                    rdiffi=AI_datasets[AI_model][key1]-AI_datasets[AI_model][key2]
                    ov_arr[key]=np.append(ov_arr[key],rdiffi)
                #
                x=ov_arr[x_str] + x_offset
                data=ov_arr[key]
                #
                params = Parameters()
                
                params.add('m', value=-np.std(data)/np.std(x))
                params.add('b', value=np.mean(data))
                #
                yerr=0*x+1.0 # important that this is 1 such that the residuals are unnormalised
                out = minimize(residual, params, args=(x, data, yerr),scale_covar=scale_yerr)
                
                m=out.params['m'].value
                b=out.params['b'].value
                resid=out.residual
                redchi=out.redchi
                db=np.std(resid)
                covar=out.covar
                
                results={'I':b,'dI':db,'residual':resid,'redchisq':redchi,'m':m,'covar':covar,'x_str':x_str}
                results_dict[key] = results
    
    else:
        for y_str in y_strs:
            key = y_str
            ov_arr[key]=np.array([])
            for AI_model in AI_datasets:
                yi=AI_datasets[AI_model][key]
                ov_arr[key]=np.append(ov_arr[key],yi)
            #
            x=ov_arr[x_str] + x_offset
            data=ov_arr[key]
            #
            params = Parameters()
            
            params.add('m', value=-np.std(data)/np.std(x))
            params.add('b', value=np.mean(data))
            #
            yerr=0*x+1.0 # important that this is 1 such that the residuals are unnormalised
            out = minimize(residual, params, args=(x, data, yerr),scale_covar=scale_yerr)
            
            m=out.params['m'].value
            b=out.params['b'].value
            resid=out.residual
            redchi=out.redchi
            db=np.std(resid)
            covar=out.covar
            
            results={'I':b,'dI':db,'residual':resid,'redchisq':redchi,'m':m,'covar':covar,'x_str':x_str}
            results_dict[key] = results
    
    return results_dict, ov_arr