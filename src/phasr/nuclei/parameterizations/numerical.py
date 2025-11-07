from ... import constants
from ..base import nucleus_base

import numpy as np
pi = np.pi

from scipy.integrate import quad
from scipy.special import spherical_jn

from ...utility import calc_and_spline
from ...utility.continuer import highenergy_continuation_exp, highenergy_continuation_poly
from ...utility.math import optimise_radius_highenergy_continuation
from ...utility.math import derivative as deriv
from ...utility.math import radial_laplace

from functools import partial

class nucleus_num(nucleus_base):
    def __init__(self,name,Z,A,rrange=[0.,20.,0.02], qrange=[0.,1000.,1.], renew=False,**args): #,R_cut=None,rho_cut=None
        nucleus_base.__init__(self,name,Z,A,**args)
        self.nucleus_type = "numerical"
        self.rrange=rrange #fm
        self.qrange=qrange #MeV
        self.renew=renew # overwrite existing calculations 
        
        if 'charge_density' in args:
             self.charge_density =  args['charge_density']
        if 'electric_field' in args:
             self.electric_field =  args['electric_field']
        if 'electric_potential' in args:
             self.electric_potential =  args['electric_potential']
        if 'form_factor' in args:
             self.form_factor =  args['form_factor']
        
        if 'weak_density' in args:
             self.weak_density =  args['weak_density']
        if 'weak_potential' in args:
            self.weak_potential =  args['weak_potential']
        
        self.update_dependencies()
        self.set_scalars_from_rho()
        
    def update_dependencies(self):
        nucleus_base.update_dependencies(self)
        #
        if hasattr(self,'total_charge'):
            if np.abs(self.total_charge - self.Z)/self.Z>1e-3:
                print('Warning total charge for '+self.name+' deviates more than 1e-3: Z='+str(self.Z)+', Q(num)='+str(self.total_charge))
        #
        if hasattr(self,'weak_charge'):
            if np.abs(self.weak_charge - self.Qw)/self.Qw>1e-3:
                print('Warning weak_charge for '+self.name+' deviates more than 1e-3: Qw='+str(self.Qw)+', Qw(num)='+str(self.weak_charge))
        #
        if hasattr(self,'electric_potential') and (not hasattr(self,'Vmin')):
            self.set_Vmin()
        
    def update_rrange(self,rrange):
        self.rrange=rrange
        self.update_dependencies()
    
    def update_qrange(self,qrange):
        self.qrange=qrange
        self.update_dependencies()

    def update_renew(self,renew):
        self.renew=renew
        #self.update_dependencies()
    
    def set_scalars_from_rho(self):

        # TODO add save and load option

        if hasattr(self,'charge_density'):
            #if not hasattr(self,"total_charge"):
            self.set_total_charge()
            #if (not hasattr(self,"charge_radius")) or (not hasattr(self,"charge_radius_sq")):
            self.set_charge_radius()
            if (hasattr(self,'k_barrett') and hasattr(self,'alpha_barrett')):# and (not hasattr(self,"barrett_moment")):
                self.set_barrett_moment()
        if hasattr(self,'rhoM0p'):# and ((not hasattr(self,'proton_radius')) or (not hasattr(self,'proton_radius_sq'))):
            self.set_proton_radius()
        if hasattr(self,'rhoM0n'):# and ((not hasattr(self,'neutron_radius')) or (not hasattr(self,'neutron_radius_sq'))):
            self.set_neutron_radius()
        if hasattr(self,'weak_density'):# and ((not hasattr(self,'weak_radius')) or (not hasattr(self,'weak_radius_sq'))):
            self.set_weak_charge()
            self.set_weak_radius()
        
        self.update_dependencies()

    def set_total_charge(self):
        self.total_charge=calc_charge(self.charge_density,self.rrange)

    def set_weak_charge(self):
        self.weak_charge=calc_charge(self.weak_density,self.rrange)

    def set_charge_radius(self,norm=None):
        if norm is None:
            norm=self.total_charge
        self.charge_radius_sq, self.charge_radius = calc_radius(self.charge_density,self.rrange,norm)
   
    def set_proton_radius(self,norm=None):
        if norm is None:
            norm=self.Z
        self.proton_radius_sq, self.proton_radius = calc_radius(self.proton_density,self.rrange,norm)
   
    def set_neutron_radius(self,norm=None):
        if norm is None:
            norm=self.A-self.Z
        self.neutron_radius_sq, self.neutron_radius = calc_radius(self.neutron_density,self.rrange,norm)
 
    def set_weak_radius(self,norm=None):
        if norm is None:
            norm=self.weak_charge
        self.weak_radius_sq, self.weak_radius = calc_radius(self.weak_density,self.rrange,norm)
    
    def set_Vmin(self):
        self.Vmin = np.min(self.electric_potential(np.arange(*self.rrange)))
    
    def barrett_moment(self,k_barrett,alpha_barrett,norm=None):
        if norm is None:
            norm=self.total_charge
        return calc_barrett_moment(self.charge_density,self.rrange,k_barrett,alpha_barrett,norm)
    
    def set_electric_field_from_charge_density(self):
        #
        def electric_field_0(r,rho=self.charge_density):
            charge_r = quad_seperator(lambda x: (x**2)*rho(x),[0,r])
            return np.sqrt(4*pi*constants.alpha_el)/(r**2)*charge_r if r!=0. else 0.  # as long as rho(0) finite follows E(0)=0, # e=np.sqrt(4*pi*alpha_el)
        # vectorize
        electric_field_vec = np.vectorize(electric_field_0)
        # spline
        electric_field_spl = spline_field(electric_field_vec,"electric_field",self.name,rrange=self.rrange,renew=self.renew)
        # highenery continue
        self.electric_field = partial(field_ultimate_poly,R=self.rrange[1]*0.95,n=2,field_spl=electric_field_spl) # Asymptotic: 1/r^2
        
    def set_electric_potential_from_electric_field(self):
        #
        Rs0 = range_seperator(self.rrange,self.electric_field)
        def electric_potential_0(r,El=self.electric_field):
            Rs=np.array([r,*Rs0[Rs0>r]])
            potential_r = quad_seperator(El,Rs)
            return - np.sqrt(4*pi*constants.alpha_el)*potential_r # e=np.sqrt(4*pi*alpha_el)
        # vectorize
        electric_potential_vec = np.vectorize(electric_potential_0)
        # spline
        electric_potential_spl = spline_field(electric_potential_vec,"electric_potential",self.name,rrange=self.rrange,renew=self.renew)
        # highenery continue
        self.electric_potential = partial(field_ultimate_poly,R=self.rrange[1]*0.95*0.95,n=1,field_spl=electric_potential_spl) # Asymptotic: 1/r
        
    def set_charge_density_from_electric_field(self):
        
        El = self.electric_field
        d_El = deriv(El,1e-6)
        
        def charge_density_vec(r,El=El,d_El=d_El):
            thresh=1e-3
            rho0 = 1/np.sqrt(4*pi*constants.alpha_el)*3*d_El(0)
            r_arr = np.atleast_1d(r)
            rho=r_arr*0+rho0
            r_mask = np.where(r_arr>=thresh) 
            if np.any(r_arr>=thresh):
                rho[r_mask] = 1/np.sqrt(4*pi*constants.alpha_el)*((2/r_arr[r_mask])*El(r_arr[r_mask]) +  d_El(r_arr[r_mask]))
            if np.isscalar(r):
                rho=rho[0]
            return rho
        
        charge_density_spl = spline_field(charge_density_vec,"charge_density",self.name,rrange=self.rrange,renew=self.renew)
        #
        # TODO test:
        r_crit = optimise_radius_highenergy_continuation(charge_density_spl,self.rrange[1],1e-3)
        #
        self.charge_density = partial(field_ultimate_exp,R=r_crit,val=0,t=0,field_spl=charge_density_spl) # Asymptotic: exp(-r)
        
    def set_electric_field_from_electric_potential(self):
        
        d_V = deriv(self.electric_potential,1e-6)
        
        def electric_field_vec(r,d_V=d_V):
            return 1/np.sqrt(4*pi*constants.alpha_el) * d_V(r)
        
        electric_field_spl = spline_field(electric_field_vec,"electric_field",self.name,rrange=self.rrange,renew=self.renew)
        # highenery continue
        self.electric_field = partial(field_ultimate_poly,R=self.rrange[1]*0.95,n=2,field_spl=electric_field_spl) # Asymptotic: 1/r^2
        
    def set_form_factor_from_charge_density(self):
        if not hasattr(self,'total_charge'):
            self.set_total_charge()
        self.form_factor = fourier_transform_pos_to_mom(self.charge_density,self.name,self.rrange,self.qrange,L=0,norm=self.total_charge,renew=self.renew)
 
    def set_charge_density_from_form_factor(self):
        #
        # problematic if FF has difficult/oscillatory highenergy behaviour. 
        #
        self.charge_density = fourier_transform_mom_to_pos(self.form_factor,self.name,self.qrange,self.rrange,L=0,norm=self.Z,renew=self.renew)
    
    def set_density_dict_from_form_factor_dict(self):
        for L in np.arange(0,2*self.spin+1,2,dtype=int):
            multipoles = [S+str(L)+nuc for S in ['M','Phipp'] for nuc in ['p','n']]
            for multipole in multipoles:
                if hasattr(self,'F'+multipole):
                    FF = getattr(self,'F'+multipole)
                    rho = fourier_transform_mom_to_pos(FF,multipole+'_'+self.name,self.qrange,self.rrange,L=L,norm=1,renew=self.renew)
                    setattr(self,'rho'+multipole,rho)
                    # 
                    if L==0:
                        rho2_vec  = partial(rho2_correction,rho0=rho)
                        # high energy continuation is very unstable before the high energy of rho sets in, hence we set the cutoff for r>rcrit 
                        rrange_laplace = [self.rrange[0],1.1*self.rrange[1],self.rrange[2]]
                        rho2_spl = spline_field(rho2_vec,"charge_density_laplace_"+multipole,self.name,rrange=rrange_laplace,renew=self.renew)
                        r_crit = optimise_radius_highenergy_continuation(rho2_spl,1.05*self.rrange[1],1e-3)
                        rho2 = partial(field_ultimate_exp,R=r_crit,val=0,t=0,field_spl=rho2_spl) # Asymptotic: exp(-r)
                        setattr(self,'rho2'+multipole,rho2)
                    #
        self.update_dependencies()

    def set_form_factor_dict_from_density_dict(self):
        for L in np.arange(0,2*self.spin+1,2,dtype=int):
            multipoles = [S+str(L)+nuc for S in ['M','Phipp'] for nuc in ['p','n']]
            for multipole in multipoles:
                if hasattr(self,'rho'+multipole):
                    rho = getattr(self,'rho'+multipole)
                    FF = fourier_transform_pos_to_mom(rho,multipole+'_'+self.name,self.rrange,self.qrange,L=L,norm=1,renew=self.renew)
                    setattr(self,'F'+multipole,FF)
        self.update_dependencies()
    
    def fill_gaps(self):
        
        if not hasattr(self,"charge_density"):
            if hasattr(self,"electric_field"):
                self.set_charge_density_from_electric_field()
            elif hasattr(self,"electric_potential"):
                self.set_electric_field_from_electric_potential()
                self.set_charge_density_from_electric_field()
            elif hasattr(self,"form_factor"):
                self.set_charge_density_from_form_factor()
            else:
                raise ValueError("Need at least one input out of charge_density, electric_field, electric_potential and form_factor to deduce the others")
        
        if not hasattr(self,"electric_field"):
            self.set_electric_field_from_charge_density()

        if not hasattr(self,"electric_potential"):
            self.set_electric_potential_from_electric_field()
        
        if not hasattr(self,"form_factor"):
            self.set_form_factor_from_charge_density()

def calc_charge(density,rrange):
    Rs = range_seperator(rrange,density)
    integral_Q = quad_seperator(lambda x: (x**2)*density(x),Rs)
    Q = 4*pi*integral_Q
    return Q

def calc_radius(density,rrange,norm):
    Rs = range_seperator(rrange,density)
    if norm==0:
        radius=np.inf
        radius_sq=np.inf
    else:
        integral_rsq = quad_seperator(lambda x: (x**4)*density(x),Rs)
        radius_sq = 4*pi*integral_rsq/norm
        radius = np.sqrt(radius_sq) if radius_sq>=0 else np.sqrt(radius_sq+0j)
    return radius_sq, radius

def calc_barrett_moment(density,rrange,k_barrett,alpha_barrett,norm):
    Rs = range_seperator(rrange,density)
    if norm==0:
        barrett=np.inf
    else:
        integral_barrett = quad_seperator(lambda x: (x**(2+k_barrett))*np.exp(-alpha_barrett*x)*density(x),Rs)
        barrett = 4*pi*integral_barrett/norm
    return barrett

def fourier_transform_pos_to_mom(fct_r,name,rrange,qrange,L=0,norm=1,renew=False):
    # r [fm] -> q [MeV]
    #
    Rs = range_seperator(rrange,fct_r)
    #
    def fct_q_0(q,rho=fct_r):
        form_factor_int = quad_seperator(lambda r: (r**2)*rho(r)*spherical_jn(L,q/constants.hc*r),Rs)
        return 4*pi*form_factor_int/norm
    # vectorize
    fct_q_vec = np.vectorize(fct_q_0)
    # spline
    fct_q_spl = spline_field(fct_q_vec,"form_factor",name,qrange,renew=renew)
    # highenery cut off at qmax
    fct_q = partial(field_ultimate_cutoff,R=qrange[1],val=0,field_spl=fct_q_spl) # Asymptotic: cutoff to 0
    #
    return fct_q

def fourier_transform_mom_to_pos(fct_q,name,qrange,rrange,L=0,norm=1,renew=False):
    # q [MeV] -> r [fm]
    #
    Qs = range_seperator(qrange,fct_q)
    #
    def fct_r_0(r,ff=fct_q): #use Z here b/c total_charge is not known b/c rho is not known
        rho_int=quad_seperator(lambda q: (q**2)*ff(q)*spherical_jn(L,r/constants.hc*q)/constants.hc**3,Qs) 
        return 4*pi*rho_int*norm/(2*pi)**3
    # vectorize
    fct_r_vec = np.vectorize(fct_r_0)
    # spline
    fct_r_spl = spline_field(fct_r_vec,"charge_density",name,rrange,renew=renew)
    #
    r_crit = optimise_radius_highenergy_continuation(fct_r_spl,rrange[1],1e-3) # set xmin to radius
    # highenergy exponential decay for rho
    fct_r = partial(field_ultimate_exp,R=r_crit,val=0,t=0,field_spl=fct_r_spl) # Asymptotic: exp(-r)
    #fct_r = partial(field_ultimate_exp,R=rrange[1],val=0,t=0,field_spl=fct_r_spl) # alternative    
    #
    return fct_r

def rho2_correction(r,rho0):
    return -radial_laplace(rho0)(r)

def range_seperator(xrange,fct):
    Xmin_int=xrange[0]
    if fct(xrange[1]+xrange[2])==0:
        Xmax_int=xrange[1]
        return np.array([Xmin_int, Xmax_int])
    else:
        Xmax_int=np.inf
        Xsep_int=xrange[1]
        return np.array([Xmin_int, Xsep_int, Xmax_int])

def quad_seperator(integrand,Rs):
    # Splits the integral according to Rs
    integral = 0
    for i in range(len(Rs)-1):
        Rmin = Rs[i]
        Rmax = Rs[i+1]
        integrali = quad(integrand,Rmin,Rmax,limit=1000)[0]
        integral += integrali 
    return integral

def spline_field(field,fieldtype,name,rrange,renew):
    field_spl=calc_and_spline(field, rrange, fieldtype+"_"+name,dtype=float,renew=renew)
    return field_spl

def field_ultimate_poly(r,R,n,field_spl): #highenergycont_field
    E_crit=field_spl(R)
    r_arr = np.atleast_1d(r)
    field=np.zeros(len(r_arr))
    mask_r = r_arr<=R
    if np.any(mask_r):
        field[mask_r] = field_spl(r_arr[mask_r])
    if np.any(~mask_r):
        field[~mask_r] = highenergy_continuation_poly(r_arr[~mask_r],R,E_crit,0,n=n)
    if np.isscalar(r):
        field=field[0]
    return field

def field_ultimate_exp(r,R,val,t,field_spl): # highenergycont_rho, often val=0, t=0
    E_crit=field_spl(R)
    dE=deriv(field_spl,1e-6)
    dE_crit=dE(R)
    r_arr = np.atleast_1d(r)
    field=np.zeros(len(r_arr))
    mask_r = r_arr<=R
    if np.any(mask_r):
        field[mask_r] = field_spl(r_arr[mask_r])
    if np.any(~mask_r):
        field[~mask_r] = highenergy_continuation_exp(r_arr[~mask_r],R,E_crit,dE_crit,val,t=t)
    if np.isscalar(r):
        field=field[0]
    return field

def field_ultimate_cutoff(r,R,val,field_spl):#  often val=0, also val=np.nan possible
    r_arr = np.atleast_1d(r)
    field=np.zeros(len(r_arr))
    mask_r = r_arr<=R
    if np.any(mask_r):
        field[mask_r] = field_spl(r_arr[mask_r])
    if np.any(~mask_r):
        field[~mask_r] = val
    if np.isscalar(r):
        field=field[0]
    return field
