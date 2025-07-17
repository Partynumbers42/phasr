import numpy as np
pi = np.pi

from scipy.special import sici

from ..nuclei import load_reference_nucleus, nucleus

class initializer():
    
    def __init__(self,Z:int,A:int,R:float,N:int,ai=None):
        
        self.Z = Z
        self.A = A
        
        self.R = R
        self.N = N
        
        if ai is None:
            self.ref_index=0
            self.set_nucleus_from_reference()
        else:
            self.ai = np.zeros(self.N)
            self.ai[:min(self.N,len(ai))] = ai[:min(self.N,len(ai))]
            self.nucleus = nucleus(name="initialized_nucleus_Z"+str(Z)+"_A"+str(A),Z=self.Z,A=self.A,ai=self.ai,R=self.R)
        
    def set_nucleus_from_reference(self):
        
        nuclei_reference =  load_reference_nucleus(self.Z,self.A)
        self.number_of_references = len(nuclei_reference)
        
        if self.number_of_references>1:    
            nucleus_reference = nuclei_reference
        else:
            nucleus_reference = nuclei_reference[self.ref_index]
        
        R_reference = nucleus_reference.R
        N_reference = nucleus_reference.N_a
        ai_reference = nucleus_reference.ai
        
        self.ai = np.zeros(self.N)
        self.ai[:min(self.N,N_reference)] = ai_reference[:min(self.N,N_reference)]
        
        # <------------------------------------ continue here
        
        if self.R == self.R_reference:
            self.ai = ai_reference
        
        
    
    def cycle_references(self):
        self.ref_index = (self.ref_index + 1) % self.number_of_references
        self.set_nucleus_from_reference()
    
def transformation_factor_ai(i:int,R_target:float,R_source:float):
    
    # numerical calculation was replaced by analytical result
    #I1=quad(lambda r: spherical_jn(0,pi*nu*r/R)*spherical_jn(0,pi*nu*r/R_ref),0,min(R,R_ref),limit=1000)
    #I2=quad(lambda r: spherical_jn(0,pi*nu*r/R)**2,0,R,limit=1000)
    #scale_factor = I1[0]/I2[0]
    
    R_max = max(R_target,R_source) 
    R_sum = R_target + R_source
    R_dif = R_target - R_source
    
    return (R_sum*sici(i*pi*R_sum/R_max)[0] - R_dif*sici(i*pi*R_dif/R_max)[0])/(2*R_target*sici(2*i*pi)[0]) if i!=0 else 1