import numpy as np
pi = np.pi

from scipy.special import sici

from ..nuclei import load_reference_nucleus, nucleus

from .parameters import ai_abs_bounds_default

class initializer():
    
    def __init__(self,Z:int,A:int,R:float,N:int,ai=None,ai_abs_bound=None):
        
        self.Z = Z
        self.A = A
        
        self.R = R
        self.N = N
        
        if ai_abs_bound is None:
            self.ai_abs_bound = ai_abs_bounds_default(np.arange(1,self.N+1),self.R,self.Z)
        else:
            self.ai_abs_bound = ai_abs_bound
        
        if ai is None:
            
            # TODO add option to load previous fit results
            
            self.ref_index=0
            self.set_ai_from_reference()
        else:
            self.ai = np.zeros(self.N)
            self.ai[:min(self.N,len(ai))] = ai[:min(self.N,len(ai))]
        
        self.overwrite_aN_from_total_charge_if_sensible()
        
        self.nucleus = nucleus(name="initialized_nucleus_Z"+str(self.Z)+"_A"+str(self.A),Z=self.Z,A=self.A,ai=self.ai,R=self.R)
    
    def set_ai_from_reference(self):
        
        nuclei_references = load_reference_nucleus(self.Z,self.A)
        self.number_of_references = len(nuclei_references)
        
        if self.number_of_references>1:    
            nucleus_reference = nuclei_references[self.ref_index]
        else:
            nucleus_reference = nuclei_references
        
        R_reference = nucleus_reference.R
        N_reference = nucleus_reference.N_a
        ai_reference = nucleus_reference.ai
        
        if self.R != R_reference:
            # guess for ai based on R
            ai_reference*=transformation_factor_ai(np.arange(1,N_reference+1),self.R,R_reference)
        
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
        self.set_ai_from_reference()
        self.overwrite_aN_from_total_charge_if_sensible()
        self.update_nucleus_ai()

def aN_from_total_charge(N,total_charge,ai,R):
    ''' only the first N-1 elements of ai are used'''
    i=np.arange(1,N)
    return -(-1)**N*((N*pi/R)**2)*( total_charge/(4*pi*R) + np.sum((-1)**i*ai[:N]/(i*pi/R)**2) )


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

