from ... import constants
from ..base import nucleus_base
from .numerical import nucleus_num

import numpy as np
pi = np.pi

from mpmath import polylog,lerchphi

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
        self.polylog2 = polylog(2,-np.exp(self.c/self.z))
        self.polylog3 = polylog(3,-np.exp(self.c/self.z))
        self.polylog5 = polylog(5,-np.exp(self.c/self.z))
        self.polylog7 = polylog(7,-np.exp(self.c/self.z))
        #
        self.nucleus_num = nucleus_num(name+self.nucleus_type,Z=Z,A=A,charge_density=self.charge_density) 
        #
        self.update_dependencies()

    def update_dependencies(self):
        nucleus_base.update_dependencies(self)
        self.charge_radius_sq = charge_radius_sq_fermi(self.c,self.z,self.w,self.polylog3,self.polylog5,self.polylog7)
        self.charge_radius = np.sqrt(self.charge_radius_sq) if self.charge_radius_sq>=0 else np.sqrt(self.charge_radius_sq+0j)
        #electric_potential_V0_fermi(self.z,self.polylog2,self.polylog3,self.total_charge)
        
    def set_form_factor_from_charge_density(self):
        self.nucleus_num.set_form_factor_from_charge_density()
        self.form_factor = self.nucleus_num.form_factor
    
    def fill_gaps(self):
        self.nucleus_num.fill_gaps()
        self.form_factor = self.nucleus_num.form_factor
        self.electric_potential = self.nucleus_num.electric_potential
        self.electric_field = self.nucleus_num.electric_field
        self.Vmin = self.nucleus_num.Vmin
        self.update_dependencies()

    def charge_density(self,r):
        return charge_density_fermi(r,self.c,self.z,self.w,self.polylog3,self.polylog5,self.total_charge)
    
    def electric_field_ana(self,r):
        print('warning: analytical field not as precise')
        return electric_field_fermi(r,self.c,self.z,self.w,self.polylog3,self.polylog5,self.total_charge)
    
    def electric_potential_ana(self,r):
        # wrong for some reason 
        print('warning: analytical potential is wrong')
        return electric_potential_fermi(r,self.c,self.z,self.w,self.polylog2,self.polylog3,self.polylog5,self.total_charge)
    
    # too slow (-> find better implementation/representation)
    def form_factor_ana(self,r):
         return form_factor_fermi(r,self.c,self.z,self.w)

def charge_density_fermi(r,c,z,w,polylog3,polylog5,Z):
    rho0 = Z/(4*pi*float(-2*z**3*polylog3-24*w*z**5*polylog5/c**2))
    return rho0*(1+ w*r**2/c**2)*np.exp(-(r-c)/z)/(1+np.exp(-(r-c)/z))

def charge_radius_sq_fermi(c,z,w,polylog3,polylog5,polylog7):
    return float(12*(c**2*z**2*polylog5+30*w*z**4*polylog7)/(c**2*polylog3+12*w*z**2*polylog5))

# do not use these:

def electric_potential_V0_fermi(z,polylog2,polylog3,Z,alpha_el=constants.alpha_el):
    # not correct
    return float(-Z*alpha_el*polylog2/(2*z*polylog3))

def electric_field_fermi_scalar(r,c,z,w,polylog3,polylog5,Z,alpha_el=constants.alpha_el):
    # something wrong with this and slow (faster with fp, but still wrong)
    poly1r = np.log(1 + np.exp((c - r)/z))
    poly2r = polylog(2,-np.exp((c - r)/z))
    poly3r = polylog(3,-np.exp((c - r)/z))
    poly4r = polylog(4,-np.exp((c - r)/z))
    poly5r = polylog(5,-np.exp((c - r)/z))
    poly3 = polylog3
    poly5 = polylog5 
    return Z*np.sqrt(alpha_el)/np.sqrt(4*pi)*float(poly1r*r**2*(c**2 + r**2*w) - 2*poly2r*r*(c**2 + 2*r**2*w)*z + 2*z**2*(c**2*poly3 - poly3r*(c**2 + 6*r**2*w) - 12*w*z*(poly4r*r - poly5*z + poly5r*z)))/float(2.*r**2*z**2*(c**2*poly3 + 12*poly5*w*z**2))
electric_field_fermi = np.vectorize(electric_field_fermi_scalar,excluded=[1,2,3,4,5,6,7])

def electric_potential_fermi_scalar(r,c,z,w,polylog2,polylog3,polylog5,Z,alpha_el=constants.alpha_el): 
    # something very wrong with this and slow (faster with fp, but still wrong)
    poly1r = np.log(1 + np.exp((c - r)/z))
    poly2r = polylog(2,-np.exp((c - r)/z))
    poly3r = polylog(3,-np.exp((c - r)/z))
    poly4r = polylog(4,-np.exp((c - r)/z))
    poly5r = polylog(5,-np.exp((c - r)/z))
    poly3 = polylog3
    poly5 = polylog5
    if r==0:
        return electric_potential_V0_fermi(z,polylog2,polylog3,Z,alpha_el)
    else:
        return Z*alpha_el*float(-2*c**2*poly3**2*z**2 + 12*poly5*r*w*z**2*(poly1r*r - poly2r*z) + poly3*(-(poly1r*r**4*w) + poly2r*r*(c**2 + 4*r**2*w)*z + 2*poly3r*(c**2 + 6*r**2*w)*z**2 + 24*w*z**3*(poly4r*r + (-poly5 + poly5r)*z)))/float(2.*poly3*r*z**2*(c**2*poly3 + 12*poly5*w*z**2))     
electric_potential_fermi = np.vectorize(electric_potential_fermi_scalar,excluded=[1,2,3,4,5,6,7,8])

def form_factor_fermi_scalar(q,c,z,w):
    # way to slow (lerchphi), unclear if correct
    q=q/constants.hc
    lp2 = np.real(1j*( lerchphi(-np.exp(c/z),2,1-1j*q*z) - lerchphi(-np.exp(c/z),2,1+1j*q*z) ))
    lp4 = np.real(1j*( lerchphi(-np.exp(c/z),4,1-1j*q*z) - lerchphi(-np.exp(c/z),4,1+1j*q*z) ))
    return float(np.exp(c/z)*(c**2*lp2 + 6*w*z**2*lp4)/(4*q*z*(c**2*polylog(3,-np.exp(c/z))+12*w*z**2*polylog(5,-np.exp(c/z)))))
form_factor_fermi = np.vectorize(form_factor_fermi_scalar,excluded=[1,2,3])