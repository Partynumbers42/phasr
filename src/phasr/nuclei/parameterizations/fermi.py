from ... import constants
from ..base import nucleus_base
from .numerical import nucleus_num

import numpy as np
pi = np.pi

from mpmath import polylog,lerchphi,fp

class nucleus_fermi(nucleus_base):
    def __init__(self,name,Z,A,c,z,**args): 
        nucleus_base.__init__(self,name,Z,A,**args)
        self.nucleus_type = "fermi"
        self.c=c
        self.z=z
        if "w" in args:
            self.w=args["w"]
            self.nucleus_type+="3p"
        else:
            self.w=0
            self.nucleus_type+="2p"
        self.total_charge=Z
        #
        self.nucleus_num = nucleus_num(name+self.nucleus_type,Z=Z,A=A,charge_density=self.charge_density,electric_potential=self.electric_potential,electric_field=self.electric_field)
        #
        self.update_dependencies()

    def update_dependencies(self):
        nucleus_base.update_dependencies(self)
        self.charge_radius_sq = charge_radius_sq_fermi(self.c,self.z,self.w)
        self.charge_radius = np.sqrt(self.charge_radius_sq) if self.charge_radius_sq>=0 else np.sqrt(self.charge_radius_sq+0j)
        self.Vmin = electric_potential_V0_FB(self.c,self.z,self.w,self.total_charge)
        
    def set_form_factor_from_charge_density(self):
        self.nucleus_num.set_form_factor_from_charge_density()
        self.form_factor = self.nucleus_num.form_factor
    
    def fill_gaps(self):
        self.nucleus_num.fill_gaps()
        self.form_factor = self.nucleus_num.form_factor
        self.update_dependencies()

    def charge_density(self,r):
        return charge_density_fermi(r,self.c,self.z,self.w,self.total_charge)
    
    def electric_field(self,r):
        return electric_field_fermi(r,self.c,self.z,self.w,self.total_charge)
    
    def electric_potential(self,r):
        return electric_potential_fermi(r,self.c,self.z,self.w,self.total_charge)
    
    #def form_factor(self,r):
    # too slow (-> find better implementation/representation)
    #     return form_factor_fermi(r,self.c,self.z,self.w)

def charge_density_fermi(r,c,z,w,Z):
    rho0 = Z/(4*pi*(-2*z**3*fp.polylog(3,-np.exp(c/z))-24*w*z**5*fp.polylog(5,-np.exp(c/z))/c**2))
    return rho0*(1+ w*r**2/c**2)*np.exp(-(r-c)/z)/(1+np.exp(-(r-c)/z))

def charge_radius_sq_fermi(c,z,w):
    return 12*(c**2*z**2*fp.polylog(5,-np.exp(c/z))+30*w*z**4*fp.polylog(7,-np.exp(c/z)))/(c**2*fp.polylog(3,-np.exp(c/z))+12*w*z**2*fp.polylog(5,-np.exp(c/z)))

def electric_potential_V0_FB(c,z,w,Z,alpha_el=constants.alpha_el):
    poly2 = fp.polylog(2,-np.exp(c/z))
    poly3 = fp.polylog(3,-np.exp(c/z))
    return -Z*alpha_el*poly2/(2*z*poly3)
    
def electric_potential_fermi_scalar(r,c,z,w,Z,alpha_el=constants.alpha_el): 
    poly1r = np.log(1 + np.exp((c - r)/z))
    poly2r = fp.polylog(2,-np.exp((c - r)/z))
    poly3r = fp.polylog(3,-np.exp((c - r)/z))
    poly4r = fp.polylog(4,-np.exp((c - r)/z))
    poly5r = fp.polylog(5,-np.exp((c - r)/z))
    poly3 = fp.polylog(3,-np.exp(c/z))
    poly5 = fp.polylog(5,-np.exp(c/z))
    if r==0:
        return electric_potential_V0_FB(c,z,w,Z,alpha_el)
    else:
        return Z*alpha_el*(-2*c**2*poly3**2*z**2 + 12*poly5*r*w*z**2*(poly1r*r - poly2r*z) + poly3*(-(poly1r*r**4*w) + poly2r*r*(c**2 + 4*r**2*w)*z + 2*poly3r*(c**2 + 6*r**2*w)*z**2 + 24*w*z**3*(poly4r*r + (-poly5 + poly5r)*z)))/(2.*poly3*r*z**2*(c**2*poly3 + 12*poly5*w*z**2)) 
electric_potential_fermi = np.vectorize(electric_potential_fermi_scalar,excluded=[1,2,3,4,5])

def electric_field_fermi_scalar(r,c,z,w,Z,alpha_el=constants.alpha_el):
    
    poly1r = np.log(1 + np.exp((c - r)/z))
    poly2r = fp.polylog(2,-np.exp((c - r)/z))
    poly3r = fp.polylog(3,-np.exp((c - r)/z))
    poly4r = fp.polylog(4,-np.exp((c - r)/z))
    poly5r = fp.polylog(5,-np.exp((c - r)/z))
    poly3 = fp.polylog(3,-np.exp(c/z))
    poly5 = fp.polylog(5,-np.exp(c/z))
    
    return Z*np.sqrt(alpha_el)/np.sqrt(4*pi)*(poly1r*r**2*(c**2 + r**2*w) - 2*poly2r*r*(c**2 + 2*r**2*w)*z + 2*z**2*(c**2*poly3 - poly3r*(c**2 + 6*r**2*w) - 12*w*z*(poly4r*r - poly5*z + poly5r*z)))/(2.*r**2*z**2*(c**2*poly3 + 12*poly5*w*z**2))
electric_field_fermi = np.vectorize(electric_field_fermi_scalar,excluded=[1,2,3,4,5])

def form_factor_fermi_scalar(q,c,z,w):
    # too slow, rewrite???
    q=q/constants.hc
    lp2 = np.real(1j*( lerchphi(-np.exp(c/z),2,1-1j*q*z) - lerchphi(-np.exp(c/z),2,1+1j*q*z) ))
    lp4 = np.real(1j*( lerchphi(-np.exp(c/z),4,1-1j*q*z) - lerchphi(-np.exp(c/z),4,1+1j*q*z) ))
    return float(np.exp(c/z)*(c**2*lp2 + 6*w*z**2*lp4)/(4*q*z*(c**2*polylog(3,-np.exp(c/z))+12*w*z**2*polylog(5,-np.exp(c/z)))))

form_factor_fermi = np.vectorize(form_factor_fermi_scalar,excluded=[1,2,3])