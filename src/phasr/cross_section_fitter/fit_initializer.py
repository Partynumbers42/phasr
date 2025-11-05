import numpy as np
pi = np.pi

from scipy.special import sici

from ..nuclei import load_reference_nucleus, nucleus

from .parameters import ai_abs_bounds_default
from .pickler import pickle_load_all_results_dicts_R

class initializer():
    
    def __init__(self,Z:int,A:int,R:float,N:int,ai=None,ai_abs_bound=None,initialize_from='reference'): #check_other_fits=False,settings={}
        
        self.Z = Z
        self.A = A
        
        self.R = R
        self.N = N
        
        if ai_abs_bound is None:
            self.ai_abs_bound = ai_abs_bounds_default(np.arange(1,self.N+1),self.R,self.Z)
        else:
            self.ai_abs_bound = ai_abs_bound
        
        if ai is None:
            if initialize_from=='reference':            
                self.ref_index=0
                print('Setting ai from dVries reference (because you asked for it)')
                self.set_ai_from_reference()
            else:
                results_dicts = pickle_load_all_results_dicts_R(self.Z,self.A,self.R,initialize_from)
                if len(results_dicts)>0:
                    print('Looking through other fits with this R:',results_dicts.keys())
                    best_p_val = 0
                    best_N_diff = np.inf
                    best_key = None
                    for results_dict_key in results_dicts:
                        current_N_diff = np.abs(results_dicts[results_dict_key]['N']-self.N)
                        if current_N_diff<best_N_diff:
                            best_N_diff = current_N_diff
                            best_p_val = 0
                        current_p_val = results_dicts[results_dict_key]['p_val']
                        if (current_N_diff==best_N_diff) and (current_p_val>=best_p_val):
                            best_key= results_dict_key
                            best_p_val = current_p_val
                    #if best_N_diff>0:
                    print('Using the dataset closest in N and with the smallest p-val: For R=',R,'N=',N,'use R=',results_dicts[best_key]['R'],', N=',results_dicts[best_key]['N'])
                    ai_best_fit = results_dicts[best_key]['ai']
                    self.ai = np.zeros(self.N)
                    self.ai[:min(self.N,len(ai_best_fit))] = ai_best_fit[:min(self.N,len(ai_best_fit))]
                    
                else:
                    self.ref_index=0
                    print('Setting ai from dVries reference (because no matching fits found)')
                    self.set_ai_from_reference()
        else:
            self.ai = np.zeros(self.N)
            self.ai[:min(self.N,len(ai))] = ai[:min(self.N,len(ai))]
        
        self.overwrite_aN_from_total_charge_if_sensible()
        
        self.nucleus = nucleus(name="initialized_nucleus_Z"+str(self.Z)+"_A"+str(self.A),Z=self.Z,A=self.A,ai=self.ai,R=self.R)
    
    def set_ai_from_reference(self):
        
        nuclei_references = load_reference_nucleus(self.Z,self.A)
        self.number_of_references = len(nuclei_references) if type(nuclei_references)==list else 1
        
        if self.number_of_references>1:    
            nucleus_reference = nuclei_references[self.ref_index]
        else:
            nucleus_reference = nuclei_references
        
        R_reference = nucleus_reference.R
        N_reference = nucleus_reference.N_a
        ai_reference = nucleus_reference.ai
        
        if self.R != R_reference:
            # guess for ai based on R 
            ni = np.arange(1,N_reference+1)
            transformation_factor_unnormalized = transformation_factor_ai(ni,self.R,R_reference)
            ai_reference*=transformation_factor_unnormalized
            qi = ni*pi/self.R
            weighting = 1
            unnormalized_weighted_total_charge = 4*pi*np.sum(-weighting*(-1)**ni*(ni*pi)*ai_reference/qi**3)
            ai_reference*=weighting*self.Z/unnormalized_weighted_total_charge
        
        self.ai = np.zeros(self.N)
        self.ai[:min(self.N,N_reference)] = ai_reference[:min(self.N,N_reference)]
    
    def overwrite_aN_from_total_charge_if_sensible(self):
        aN = aN_from_total_charge(self.N,self.Z,self.ai,self.R)
        if -self.ai_abs_bound[self.N-1]<=aN<=self.ai_abs_bound[self.N-1]:         
            self.ai[self.N-1]=aN 
        
    def update_nucleus_ai(self):
        self.nucleus.update_ai(self.ai)
            
    def cycle_references(self):
        self.ref_index = (self.ref_index + 1) % self.number_of_references
        print('select reference Nr.:',self.ref_index)
        self.set_ai_from_reference()
        self.overwrite_aN_from_total_charge_if_sensible()
        self.update_nucleus_ai()

def aN_from_total_charge(N,total_charge,ai,R):
    ''' only the first N-1 elements of ai are used'''
    ni=np.arange(1,N)
    return -(-1)**N*((N*pi/R)**2)*( total_charge/(4*pi*R) + np.sum((-1)**ni*ai[:N-1]/(ni*pi/R)**2) )


def transformation_factor_ai(ni,R_target:float,R_source:float):
    
    # numerical calculation was replaced by analytical vectorized result
    #I1=quad(lambda r: spherical_jn(0,pi*nu*r/R)*spherical_jn(0,pi*nu*r/R_ref),0,min(R,R_ref),limit=1000)
    #I2=quad(lambda r: spherical_jn(0,pi*nu*r/R)**2,0,R,limit=1000)
    #scale_factor = I1[0]/I2[0]
    
    R_max = max(R_target,R_source) 
    R_sum = R_target + R_source
    R_dif = R_target - R_source
    
    ni_vec = np.atleast_1d(ni)
    transformation_factor = np.ones(len(ni_vec))
    mask_ni = (ni!=0)
    if np.any(mask_ni):
        transformation_factor[mask_ni] = (R_sum*sici(ni[mask_ni]*pi*R_sum/R_max)[0] - R_dif*sici(ni[mask_ni]*pi*R_dif/R_max)[0])/(2*R_target*sici(2*ni[mask_ni]*pi)[0])
    if np.isscalar(ni):
        transformation_factor=transformation_factor[0]    
    return transformation_factor

