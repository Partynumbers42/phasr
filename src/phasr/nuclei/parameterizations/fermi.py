from ... import constants
from ..base import nucleus_base
from .numerical import nucleus_num

import numpy as np
pi = np.pi

from mpmath import fp, polylog#, lerchphi

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
        self.polylog2 = float(polylog(2,-np.exp(self.c/self.z)))
        self.polylog3 = float(polylog(3,-np.exp(self.c/self.z)))
        self.polylog4 = float(polylog(4,-np.exp(self.c/self.z)))
        self.polylog5 = float(polylog(5,-np.exp(self.c/self.z)))
        self.polylog7 = float(polylog(7,-np.exp(self.c/self.z)))
        self.charge_density_norm = self.total_charge/(4*pi)*1./(-2*self.z**3*self.polylog3-24*self.w*self.z**5*self.polylog5/self.c**2)
        #
        self.update_dependencies()

    def update_dependencies(self):
        #
        structures_num={}
        for structure in ['charge_density','electric_field','electric_potential','form_factor','weak_density','proton_density','neutron_density']:
            if hasattr(self,structure):
                structures_num[structure]=getattr(self,structure) 
        self.nucleus_num = nucleus_num(self.name+self.nucleus_type,Z=self.Z,A=self.A,**structures_num) 
        #
        nucleus_base.update_dependencies(self)
        self.charge_radius_sq = charge_radius_sq_fermi(self.c,self.z,self.w,self.polylog5,self.polylog7,self.charge_density_norm,self.total_charge)
        self.charge_radius = np.sqrt(self.charge_radius_sq) if self.charge_radius_sq>=0 else np.sqrt(self.charge_radius_sq+0j)
        self.Vmin_ana = electric_potential_V0_fermi(self.c,self.z,self.w,self.polylog2,self.polylog4,self.charge_density_norm)
    
    def set_form_factor_from_charge_density(self):
        self.nucleus_num.set_form_factor_from_charge_density()
        self.form_factor = self.nucleus_num.form_factor
    
    def fill_gaps(self):
        self.nucleus_num.fill_gaps()
        self.nucleus_num.update_dependencies()
        self.form_factor = self.nucleus_num.form_factor
        self.electric_potential = self.nucleus_num.electric_potential
        self.electric_field = self.nucleus_num.electric_field
        self.Vmin = self.nucleus_num.Vmin
        
        for attribute in ['weak_charge','weak_radius','weak_radius_sq','proton_density','proton_radius','proton_radius_sq','neutron_density','neutron_radius','neutron_radius_sq']:
            if hasattr(self.nucleus_num,attribute):
                setattr(self,attribute,getattr(self.nucleus_num,attribute))
        
        self.update_dependencies()

    def charge_density(self,r):
        return charge_density_fermi(r,self.c,self.z,self.w,self.charge_density_norm)
    
    def electric_field_ana(self,r):
        print('warning: analytical field not precise enough')
        return electric_field_fermi(r,self.c,self.z,self.w,self.polylog3,self.polylog5,self.charge_density_norm)
    
    def electric_potential_ana(self,r):
        print('warning: analytical potential not precise enough')
        return electric_potential_fermi(r,self.c,self.z,self.w,self.polylog2,self.polylog3,self.polylog4,self.polylog5,self.charge_density_norm)
    
    # def form_factor_ana(self,r):
    #     print('warning: analytical form factor very slow and wrong')
    #     return form_factor_fermi(r,self.c,self.z,self.w,self.charge_density_norm,self.total_charge)

def charge_density_fermi(r,c,z,w,norm):
    return norm*(1+ w*r**2/c**2)*np.exp(-(r-c)/z)/(1+np.exp(-(r-c)/z))

def charge_radius_sq_fermi(c,z,w,polylog5,polylog7,norm,Z):
    #
    poly5=polylog5
    poly7=polylog7
    return -4*pi*norm/Z*24*(poly5*z**5 + (30*poly7*w*z**7)/c**2)

def electric_potential_V0_fermi(c,z,w,polylog2,polylog4,norm,alpha_el=constants.alpha_el):
    # not as precise as numerically
    poly2=polylog2
    poly4=polylog4
    return 4*pi*alpha_el*norm*z**2*(poly2 + 6*poly4*w*z**2/c**2)

def electric_field_fermi_scalar(r,c,z,w,polylog3,polylog5,norm,alpha_el=constants.alpha_el):
    # slow and not as precise as numerically
    poly1r = np.log(1 + np.exp((c - r)/z))
    poly2r = fp.polylog(2,-np.exp((c - r)/z))
    poly3r = fp.polylog(3,-np.exp((c - r)/z))
    poly4r = fp.polylog(4,-np.exp((c - r)/z))
    poly5r = fp.polylog(5,-np.exp((c - r)/z))
    poly3 = polylog3
    poly5 = polylog5 
    return  np.sqrt(4*pi*alpha_el)*norm*(z*(-(poly1r*r**2*(c**2 + r**2*w)) + 2*z*(poly2r*r*(c**2 + 2*r**2*w) +  z*(-(c**2*poly3) + poly3r*(c**2 + 6*r**2*w) + 12*w*z*(poly4r*r - poly5*z + poly5r*z)))))/(c**2*r**2)
electric_field_fermi = np.vectorize(electric_field_fermi_scalar,excluded=[1,2,3,4,5,6,7])

def electric_potential_fermi_scalar(r,c,z,w,polylog2,polylog3,polylog4,polylog5,norm,alpha_el=constants.alpha_el): 
    # slow and not as precise as numerically
    poly2r = fp.polylog(2,-np.exp((c - r)/z))
    poly3r = fp.polylog(3,-np.exp((c - r)/z))
    poly4r = fp.polylog(4,-np.exp((c - r)/z))
    poly5r = fp.polylog(5,-np.exp((c - r)/z))
    poly3 = polylog3
    poly5 = polylog5
    if r==0:
        return electric_potential_V0_fermi(c,z,w,polylog2,polylog4,norm,alpha_el)
    else:
        return -4*pi*alpha_el*norm*z**2*(poly2r*r*(c**2 + r**2*w) + 2*z*(-(c**2*poly3) + poly3r*(c**2 + 3*r**2*w) + 3*w*z*(3*poly4r*r - 4*poly5*z + 4*poly5r*z)))/(c**2*r)
electric_potential_fermi = np.vectorize(electric_potential_fermi_scalar,excluded=[1,2,3,4,5,6,7,8,9])

# def form_factor_fermi_scalar(q,c,z,w,norm,Z):
#     # way to slow (lerchphi)
#     # wrong (b/c wrong sheet???)
#     q=q/constants.hc
#     lp2 = 1j*( lerchphi(-np.exp(c/z),2,1-1j*q*z) - lerchphi(-np.exp(c/z),2,1+1j*q*z) )
#     lp4 = 1j*( lerchphi(-np.exp(c/z),4,1-1j*q*z) - lerchphi(-np.exp(c/z),4,1+1j*q*z) )
#     return - 4*pi*norm/Z*np.exp(c/z)*z**2*float(np.real(c**2*lp2 + 6*w*z**2*lp4))/(2*c**2*q)
# form_factor_fermi = np.vectorize(form_factor_fermi_scalar,excluded=[1,2,3,4,5])