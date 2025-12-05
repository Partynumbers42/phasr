from ...config import local_paths

from ... import constants, masses

import numpy as np
pi=np.pi

import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import os

from scipy.interpolate import splev, splrep
from scipy.optimize import minimize

#import numdifftools as ndt
#from scipy.linalg import inv

from ...physical_constants.iaea_nds import massofnucleusZN, JPofnucleusZN
from ...nuclei import nucleus
from ...nuclei.parameterizations.numerical import field_ultimate_cutoff#, highenergycutoff_field

from ...utility.math import short_uncertainty_notation

from .overlap_integrals import overlap_integral_scalar, overlap_integral_vector, overlap_integral_dipole

from .left_right_asymmetry import left_right_asymmetry_lepton_nucleus_scattering

from functools import partial

# the code of this submodule is fairly specialized to the initial specific use case and should be generalized
# no guarantied for the contents and reliability of this submodule

def prepare_results(Z,A,folder_path,name=None,r_cut=None,print_radius_check=False): 
    # the code assumes a folder with two files per (ab-inito) calculation following the naming scheme of name+'.csv' and name+'_FF.csv',
    # containing the relevant scalar parameters and quantities as well as the form factors respectively 
    # 
    # r_cut needs to be accessed for the considered nucleus
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
                # this normalization adjustment is specific to the normalization of the data we used 
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

        # spline and add CMS corrections (form factors need to be corrected for center of mass effects)
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

def calculate_correlation_quantities(AI_datasets,quantities_fct_dict={},renew=False,verbose=True,verboseLoad=True):
    # Warning renew renews everything
    #
    for AI_model in AI_datasets:
        
        prekeys = list(AI_datasets[AI_model].keys())
        
        path_correlation_quantities=local_paths.correlation_quantities_paths + "correlation_quantities_"+AI_datasets[AI_model]['atom'].name+".txt"
        
        os.makedirs(os.path.dirname(path_correlation_quantities), exist_ok=True)

        if os.path.exists(path_correlation_quantities) and renew==False:
            with open( path_correlation_quantities, "rb" ) as file:
                correlation_quantities_array = np.genfromtxt( file,comments=None,skip_header=0,delimiter=',',names=['par','val'],autostrip=True,dtype=['<U100',float])
            
            saved_values = {quantity_tuple[0]:quantity_tuple[1] for quantity_tuple in correlation_quantities_array}
            AI_datasets[AI_model]={**AI_datasets[AI_model],**saved_values}
            saved_keys = list(saved_values.keys())
            
            if verboseLoad and len(saved_keys)>0:
                print("Loaded (existing) correlation quantities for "+str(AI_model)+" from ",path_correlation_quantities,": ",saved_keys)
            
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
        if (not 'rw-rch' in saved_keys) or renew:
            AI_datasets[AI_model]['rw-rch']=atom_key.weak_radius - atom_key.charge_radius
        if (not 'rn-rp' in saved_keys) or renew:
            AI_datasets[AI_model]['rn-rp']=atom_key.neutron_radius - atom_key.proton_radius
        #
        for quantity_key in quantities_fct_dict:
            if type(quantity_key) == np.ndarray or type(quantity_key) == tuple:
                # if a function has more then one return value
                sub_key_missing = False
                for quantity_sub_key in quantity_key:
                    if (not quantity_sub_key in saved_keys) or renew:
                        sub_key_missing = True
                if sub_key_missing:
                    quantities_tuple = quantities_fct_dict[quantity_key](atom_key)
                    tuple_index = 0
                    for quantity_sub_key in quantity_key:
                        AI_datasets[AI_model][quantity_sub_key] = quantities_tuple[tuple_index]
                        tuple_index+=0
            else:
                if (not quantity_key in saved_keys) or renew:
                    AI_datasets[AI_model][quantity_key] = quantities_fct_dict[quantity_key](atom_key)
            
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

def S_N(atom_key,nuc,reference_nucleus=None,**overlap_integral_args):
    if reference_nucleus == None:
        reference_nucleus = atom_key
    return overlap_integral_scalar(reference_nucleus,nuc,nucleus_response=atom_key,nonzero_electron_mass=True,**overlap_integral_args)

def V_N(atom_key,nuc,reference_nucleus=None,**overlap_integral_args):
    if reference_nucleus == None:
        reference_nucleus = atom_key
    return overlap_integral_vector(reference_nucleus,nuc,nucleus_response=atom_key,nonzero_electron_mass=True,**overlap_integral_args)

def calculate_correlation_SI_overlap_integrals(AI_datasets,reference_nucleus=None,overlap_integral_args={},**args):
        quantities_fct_dict={}
        nuc_ref_str = reference_nucleus.name if reference_nucleus is not None else 'from_dataset'
        for nuc in ['p','n','ch']:
            #key='M0'+nuc
            quantities_fct_dict['S'+nuc+'_rhoch_'+nuc_ref_str] = partial(S_N,nuc=nuc,reference_nucleus=reference_nucleus,**overlap_integral_args)
            quantities_fct_dict['V'+nuc+'_rhoch_'+nuc_ref_str] = partial(V_N,nuc=nuc,reference_nucleus=reference_nucleus,**overlap_integral_args)           
        return calculate_correlation_quantities(AI_datasets,quantities_fct_dict,**args)

def calculate_correlation_form_factors(AI_datasets,q_exp,**args):
        def Fch_q(atom_key):
            return atom_key.Fch(q_exp,L=0)
        def Fw_q(atom_key):
            return atom_key.Fw(q_exp,L=0)
        q_str = 'q{:.3f}'.format(q_exp)        
        quantities_fct_dict={'Fch_'+q_str:Fch_q,'Fw_'+q_str:Fw_q}
        return calculate_correlation_quantities(AI_datasets,quantities_fct_dict,**args)

def calculate_correlation_left_right_asymmetry(AI_datasets,E_exp,theta_exp,acceptance_exp=None,reference_nucleus=None,left_right_asymmetry_args={},**args):
        #
        nuc_ref_str = reference_nucleus.name if reference_nucleus is not None else 'from_dataset'
        #
        if acceptance_exp is None:
            def APV(atom_key):
                return left_right_asymmetry_lepton_nucleus_scattering(E_exp,theta_exp,atom_key,atom_key if reference_nucleus is None else reference_nucleus,verbose=True,**left_right_asymmetry_args)
            quantities_fct_dict={'APV_'+'E{:.2f}_theta{:.4f}'.format(E_exp,theta_exp)+'_rhoch_'+nuc_ref_str:APV}
        else:
            def APV(atom_key):
                return left_right_asymmetry_lepton_nucleus_scattering(E_exp,theta_exp,atom_key,atom_key if reference_nucleus is None else reference_nucleus,acceptance=acceptance_exp,verbose=True,**left_right_asymmetry_args)
            
            # -> not working yet needs to be reworked, dict key cant be list
            
            quantities_fct_dict={('theta_'+'E{:.2f}_weighted_mean'.format(E_exp)+'_rhoch_'+nuc_ref_str,'Qsq_'+'E{:.2f}_weighted_mean'.format(E_exp)+'_rhoch_'+nuc_ref_str,'APV_'+'E{:.2f}_weighted_mean'.format(E_exp)+'_rhoch_'+nuc_ref_str):APV}
        #
        return calculate_correlation_quantities(AI_datasets,quantities_fct_dict,**args)

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
    
    # normalize data in x and y direction 
    x_data_mean = np.mean(x_data_unnormalized)
    x_data_std = np.std(x_data_unnormalized)
    y_data_mean = np.mean(y_data_unnormalized)
    y_data_std = np.std(y_data_unnormalized)
    x_data = (x_data_unnormalized - x_data_mean) / x_data_std
    y_data = (y_data_unnormalized - y_data_mean) / y_data_std
    
    # fit normalized 
    y_error=np.std(y_data)
    m_ini = -np.std(y_data)/np.std(x_data) # somewhat arbitrary initial value
    b_ini = np.mean(y_data) # somewhat arbitrary initial value
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

def Correlator(AI_datasets,x_str='rchsq',y_strs=['rwsq'],x_offset=0,**minimizer_args): 
    #
    val_arr_dict={}
    #
    val_arr_dict[x_str]=np.array([])
    for AI_model in AI_datasets:
        rsqi=AI_datasets[AI_model][x_str]
        val_arr_dict[x_str]=np.append(val_arr_dict[x_str],rsqi)
        
    results_dict={}
    
    for y_str in y_strs:
        key = y_str
        val_arr_dict[key]=np.array([])
        for AI_model in AI_datasets:
            yi=AI_datasets[AI_model][key]
            val_arr_dict[key]=np.append(val_arr_dict[key],yi)
        results_dict[key] = fit_linear_correlation(val_arr_dict,x_str,key,x_offset,**minimizer_args)
        
    return results_dict, val_arr_dict

def implications_of_correlation(AI_datasets,x_str,y_strs,x_ref,dx_ref,verbose=True):

    x_offset = -x_ref
    results_dict, _ = Correlator(AI_datasets,x_str,y_strs,x_offset=x_offset)

    correlation_results_dict = {x_str:{'val':x_ref,'error_dr':dx_ref,'error_corr':0}}

    for y_str in y_strs:
        
        results=results_dict[y_str]
        m=results['m']
        b=results['val']
        db=results['dval']
        
        y_val=(x_ref + x_offset)*m+b # = b
        dy_dr=np.sqrt(dx_ref**2*m**2)
        dy_corr=db

        correlation_results_dict[y_str]={'val':y_val,'error_dr':dy_dr,'error_corr':dy_corr}
        
        if verbose:
            y_val_str, [dy_dr_str,dy_corr_str] = short_uncertainty_notation(y_val,[dy_dr,dy_corr])
            print(y_str+'='+y_val_str+r'('+dy_dr_str+r')('+dy_corr_str+r')')

    return correlation_results_dict


# for convenience
def rsq_dict_add_r_vals(rsq_correlation_results_dict,verbose=True):
    r_correlation_results_dict = {}
    for rsq_str in rsq_correlation_results_dict:
        if rsq_str[0]=='r':
            r_str=rsq_str[:-2]
            rsq_val=rsq_correlation_results_dict[rsq_str]['val']
            drsq_dr=rsq_correlation_results_dict[rsq_str]['error_dr']
            drsq_corr=rsq_correlation_results_dict[rsq_str]['error_corr']
            r_val = np.sqrt(rsq_val)
            dr_dr = drsq_dr/(2*r_val)
            dr_corr = drsq_corr/(2*r_val)
            r_correlation_results_dict[r_str] = {'val':r_val,'error_dr':dr_dr,'error_corr':dr_corr}

            if verbose:
                r_val_str, [dr_dr_str,dr_corr_str] = short_uncertainty_notation(r_val,[dr_dr,dr_corr])
                print(r_str+'='+r_val_str+r'('+dr_dr_str+r')('+dr_corr_str+r')')
        
    return r_correlation_results_dict


# plotting routine
def plot_correlation(ax,AI_datasets,x_str,y_strs,x_ref=None,dx_ref=None,x_ref_label=None,y_str_label_trans=lambda x: x ,xrange=None,yrange=None,plot_fit=True,plot_color_legend=True,plot_marker_legend=True,hatch=None):

    AI_names = {'DN2LOGO':r'$\Delta$NNLO$_\operatorname{GO}$','N2LOsat':r'NNLO$_\operatorname{sat}$','EM1p82p0':r'$1.8/2.0$ (EM)','EM2p02p0':r'$2.0/2.0$ (EM)','EM2p02p0PWA':r'$2.0/2.0$ (PWA)','EM2p22p0':r'$2.2/2.0$ (EM)','1p82p0EM7p5':'1.8/2.0 (EM7.5)', '1p82p0sim7p5':'1.8/2.0 (sim7.5)','NIsample':'Samples from\nHu et al. (2022)','SM':'shell model'}
    marker_dict={'DN2LOGO':'s','N2LOsat':'D','EM1p82p0':'p','EM2p02p0':'X','EM2p02p0PWA':'d','EM2p22p0':'P','1p82p0EM7p5':'v','1p82p0sim7p5':'^','NIsample':'.','SM':'h'}
    
    AI_names = { key : AI_names[key] for key in AI_names if key in str(AI_datasets.keys())}
    marker_dict = { key : marker_dict[key] for key in marker_dict if key in str(AI_datasets.keys())}
    
    x_offset = -x_ref if not x_ref is None else 0
    results_dict, val_arr_dict = Correlator(AI_datasets,x_str,y_strs,x_offset=x_offset)
    
    if xrange is None:

        if x_ref is None:
            x_extra_min = val_arr_dict[x_str][0]
            x_extra_max = val_arr_dict[x_str][0]
        elif dx_ref is None:
            x_extra_min = x_ref
            x_extra_max = x_ref
        else:
            x_extra_min = x_ref - dx_ref
            x_extra_max = x_ref + dx_ref
        
        x_val_min, x_val_max = np.min(np.append(val_arr_dict[x_str],x_extra_min)), np.max(np.append(val_arr_dict[x_str],x_extra_max)) 
        x_tics = (x_val_max-x_val_min)*2e-1
        x_min ,x_max = x_val_min - x_tics , x_val_max + x_tics
        x_round_digs=-int(np.floor(np.log10(np.abs(x_tics))))
        x_min=np.around(x_min,x_round_digs)
        x_max=np.around(x_max,x_round_digs)
        x_tics=np.around(x_tics,x_round_digs)
    else:
        x_min ,x_max, x_tics = xrange[0], xrange[1], xrange[2]
    x_bin = x_tics * 1.e-3

    if yrange is None:
        y_val_min, y_val_max = +np.inf, -np.inf
        for y_str in y_strs:
            y_val_min, y_val_max = np.min(np.append(val_arr_dict[y_str],y_val_min)), np.max(np.append(val_arr_dict[y_str],y_val_max))
        y_tics = (y_val_max-y_val_min)*2e-1
        y_min ,y_max = y_val_min - y_tics , y_val_max + y_tics
        y_round_digs=-int(np.floor(np.log10(np.abs(y_tics))))
        y_min=np.around(y_min,y_round_digs)
        y_max=np.around(y_max,y_round_digs)
        y_tics=np.around(y_tics,y_round_digs)
    else:
        y_min ,y_max, y_tics = yrange[0], yrange[1], yrange[2]
    y_bin = y_tics * 1.e-3
    
    x=np.arange(x_min,x_max+x_bin,x_bin)
    
    color_nr=0
    for y_str in y_strs:
        color_nr+=1
        if color_nr==3:
            color_nr+=1
    
        first=True
        for AI_key in AI_datasets:
            AI_name=AI_key if AI_key[:-4]!='NIsample' else 'NIsample'
            ax.scatter(AI_datasets[AI_key][x_str],AI_datasets[AI_key][y_str],marker=marker_dict[AI_name],edgecolor='black',linewidth=0.2,s=50,hatch=hatch,alpha=1,label=(y_str_label_trans(y_str) if first else None),color='C'+str(color_nr),zorder=2 if AI_name!='NIsample' else 1)
            first=False
        
    if plot_fit:
        for y_str in results_dict:
            
            results=results_dict[y_str]
            m=results['m']
            b=results['val']
            db=results['dval']
            
            y=(x + x_offset)*m+b
            ax.plot(x,y,color='grey',zorder=0,linewidth=0.5)
            ax.fill_between(x,y+db,y-db,alpha=0.25,color='grey',zorder=-1)

    if not x_ref is None: 
        ax.plot([x_ref,x_ref],[y_min,y_max],linestyle='--',color='C3',zorder=-2)
        if not dx_ref is None:
            ax.fill_betweenx([y_min,y_max],2*[x_ref-dx_ref],2*[x_ref+dx_ref],alpha=0.25,color='C3',zorder=-3,edgecolor=None)
        if not x_ref_label is None:
            ax.annotate(x_ref_label, (x_ref+x_bin,y_min+y_bin),horizontalalignment='left', verticalalignment='bottom',color='C3')
        
    ax.set_xticks(np.arange(x_min,x_max+x_tics,x_tics,dtype=type(x_tics)))#,rotation = 15
    ax.set_xlim(x_min,x_max)
    ax.set_yticks(np.arange(y_min,y_max+y_tics,y_tics,dtype=type(y_tics)))#,rotation = 15
    ax.set_ylim(y_min,y_max)
    ax.minorticks_on()

    # top
    if plot_color_legend:
        handles = [mpatches.Patch(color=line._facecolors) for line in ax.get_legend_handles_labels()[0]]
        leg1 = ax.legend(handles, ax.get_legend_handles_labels()[1], ncol=4, bbox_to_anchor=(0.00, 1.00,1.00,0), loc="upper left", fontsize=8) #,mode="expand"
        leg1.set_in_layout(True)
        ax.add_artist(leg1)
        #art1.set_in_layout(True)
    
    # bottom
    if plot_marker_legend:
        handles = [mlines.Line2D([], [], marker=marker, mec='k', mfc='w', ls='') for marker in list(marker_dict.values())]
        handles_names = list(AI_names.values())
        ax.legend(handles, handles_names, ncol=3, loc='upper left',fontsize=8, bbox_to_anchor=(0.00, -0.2, 1.00, 0),mode="expand")
        #leg2.set_in_layout(True)
        #art2 = ax.add_artist(leg2)
        #art2.set_in_layout(True)