from ... import constants
from ..base import nucleus_base

import numpy as np
pi = np.pi

from scipy.integrate import quad
from scipy.special import spherical_jn

from ...utility import calcandspline
from ...utility.continuer import highenergycontinuation_exp, highenergycontinuation_poly
from ...utility.math import derivative as deriv

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

        self.update_dependencies()
        
    def update_dependencies(self):
        nucleus_base.update_dependencies(self)
        #
        if hasattr(self,'charge_density'):
            self.set_scalars_from_rho()
            if np.abs(self.total_charge - self.Z)/self.Z>1e-4:
                print('Warning total charge for '+self.name+' deviates more than 1e-4: Z='+str(self.Z)+', Q(num)='+str(self.total_charge))
        #
        if hasattr(self,'weak_density'):
            self.set_weak_charge()
            if np.abs(self.weak_charge - self.Qw)/self.Qw>1e-4:
                print('Warning weak_charge for '+self.name+' deviates more than 1e-4: Qw='+str(self.Qw)+', Qw(num)='+str(self.weak_charge))
        #
        if hasattr(self,'electric_potential'):
            if not hasattr(self,'Vmin'):
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

    def set_barrett_moment(self,norm=None):
        if norm is None:
            norm=self.total_charge
        self.barrett_moment = calc_barrett_moment(self.charge_density,self.rrange,self.k_barrett,self.alpha_barrett,norm)

    def set_Vmin(self):
        self.Vmin = np.min(self.electric_potential(np.arange(*self.rrange)))
        
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
        self.electric_field = highenergycont_field(electric_field_spl,R=self.rrange[1]*0.95,n=2) # Asymptotic: 1/r^2

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
        self.electric_potential = highenergycont_field(electric_potential_spl,R=self.rrange[1]*0.95*0.95,n=1) # Asymptotic: 1/r

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
        # TODO Adjust r_crit=rrange[1] if rho<0,drho>0 ? beyond oscillatory 
        self.charge_density = highenergycont_rho(charge_density_spl,R=self.rrange[1],val=0,t=0)
        
    def set_electric_field_from_electric_potential(self):
        
        d_V = deriv(self.electric_potential,1e-6)
        
        def electric_field_vec(r,d_V=d_V):
            return 1/np.sqrt(4*pi*constants.alpha_el) * d_V(r)
        
        electric_field_spl = spline_field(electric_field_vec,"electric_field",self.name,rrange=self.rrange,renew=self.renew)
        # highenery continue
        self.electric_field = highenergycont_field(electric_field_spl,R=self.rrange[1]*0.95,n=2) # Asymptotic: 1/r^2

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
                    setattr(self,'rho'+multipole,fourier_transform_mom_to_pos(getattr(self,'F'+multipole),multipole+'_'+self.name,self.qrange,self.rrange,L=L,norm=1,renew=self.renew))

    def set_form_factor_dict_from_density_dict(self):
        for L in np.arange(0,2*self.spin+1,2,dtype=int):
            multipoles = [S+str(L)+nuc for S in ['M','Phipp'] for nuc in ['p','n']]
            for multipole in multipoles:
                if hasattr(self,'rho'+multipole):
                    setattr(self,'F'+multipole,fourier_transform_pos_to_mom(getattr(self,'rho'+multipole),multipole+'_'+self.name,self.rrange,self.qrange,L=L,norm=1,renew=self.renew))


    # TODO <-- clean implementation of rho_dict vs FF_dict / maybe separate, rho, E, V and FF ?
    #
    # def set_rho_dict_from_FF_dict(self):
    #     #
    #     self.wanna_calc('Charge densities (L>=0)')
    #     if not self.calc:# or self.spin==0: # reconsider this here
    #         return None
    #     #
    #     self.charge_density_dict = {}
    #     #
    #     #
    #     for L in np.arange(0,2*self.spin+1,1,dtype=int):
    #         #
    #         #print(L)
    #         #
    #         key_Fch='Fch'+str(L)+'c'
    #         key_Fmag='Fmag'+str(L)+'c'
    #         key_Fw='Fw'+str(L)+'c'
    #         key_rho='rho'+str(L)
    #         key_j='j'+str(L)+str(L)+'imag'
    #         key_rhow='rhow'+str(L)
    #         #
                
    #         if L%2==0 and (key_Fch in self.form_factor_dict):
    #             #
    #             #print('calc even L='+str(L))
    #             #
    #             if L==0 and (self.charge_density is not None):
    #                 print('set L=0 from self.charge density')
    #                 self.charge_density_dict[key_rho]=self.charge_density
    #             else:     
    #                 Fch = self.form_factor_dict[key_Fch]
    #                 #
    #                 if Fch(self.qrange[1]+self.qrange[2])==0:
    #                     Qmax_int=self.qrange[1]
    #                 else:
    #                     Qmax_int=np.inf
    #                 #
    #                 #print(Qmax_int)
    #                 #
    #                 def charge_density_0(r,FF=Fch,norm=self.total_charge):
    #                     rho_int=quad(lambda q: (q**2)*FF(q*hc)*spherical_jn(L,r*q),self.qrange[0]/hc,Qmax_int/hc,limit=1000) 
    #                     return 4*pi*rho_int[0]*norm/(2*pi)**3
    #                 # vectorize
    #                 charge_density_vec = np.vectorize(charge_density_0)
    #                 # spline
    #                 charge_density_spl = spline_field(charge_density_vec,"charge_density_L"+str(L),self.name,rrange=self.rrange,renew=self.renew)
    #                 # exponential decay for rho
    #                 self.charge_density_dict[key_rho] = highenergycont_rho(charge_density_spl,R=self.rrange[1],val=0,t=0)
    #                 #
    #                 if L==0 and (self.charge_density is None): # -> hasattr
    #                     #print('overwrite L=0 self.charge_density')
    #                     self.charge_density = copy.copy(self.charge_density_dict[key_rho])
    #             #
    #             Fw = self.form_factor_dict[key_Fw]
    #             #
    #             if Fw(self.qrange[1]+self.qrange[2])==0:
    #                 Qmax_int=self.qrange[1]
    #             else:
    #                 Qmax_int=np.inf
    #             #
    #             def weak_density_0(r,FF=Fw,norm=self.weak_charge):
    #                 rhow_int=quad(lambda q: (q**2)*FF(q*hc)*spherical_jn(L,r*q),self.qrange[0]/hc,Qmax_int/hc,limit=1000) 
    #                 return 4*pi*rhow_int[0]*norm/(2*pi)**3
    #             # vectorize
    #             weak_density_vec = np.vectorize(weak_density_0)
    #             # spline
    #             weak_density_spl = spline_field(weak_density_vec,"weak_density_L"+str(L),self.name,rrange=self.rrange,renew=self.renew)
    #             # exponential decay for rho
    #             self.charge_density_dict[key_rhow] = highenergycont_rho(weak_density_spl,R=self.rrange[1],val=0,t=0)
    #             #
    #         elif L%2==1 and (key_Fmag in self.form_factor_dict):
    #             #
    #             #print('calc odd L='+str(L))
    #             #
    #             Fmag = self.form_factor_dict[key_Fmag]
    #             #
    #             if Fmag(self.qrange[1]+self.qrange[2])==0:
    #                 Qmax_int=self.qrange[1]
    #             else:
    #                 Qmax_int=np.inf
    #             #
    #             # FF is assumed to be purely imaginary !!!
    #             def charge_current_0(r,FF=Fmag):
    #                 j_int=quad(lambda q: (q**2)*np.imag(FF(q*hc))*spherical_jn(L,r*q),self.qrange[0]/hc,Qmax_int/hc,limit=1000) 
    #                 return 4*pi*j_int[0]/(2*pi)**3
    #             # vectorize
    #             charge_current_vec = np.vectorize(charge_current_0)
    #             # spline
    #             charge_current_spl = spline_field(charge_current_vec,"charge_current_L"+str(L),self.name,rrange=self.rrange,renew=self.renew)
    #             # exponential decay for rho
    #             self.charge_density_dict[key_j] = highenergycont_rho(charge_current_spl,R=self.rrange[1],val=0,t=0)
    #             #
        
    #     for key_FF in self.multipoles_keys:
            
    #         key0=key_FF[1:] # extract name
    #         L = int(key_FF[-2]) #extract L for bessel fct
            
    #         if key0 in self.calc_multipoles:
    #             key_rho = 'rho'+key0
    #             FF = self.form_factor_dict[key_FF+'c'] # only with CMS corrections
                
    #             if FF(self.qrange[1]+self.qrange[2])==0:
    #                 Qmax_int=self.qrange[1]
    #             else:
    #                 Qmax_int=np.inf
    #             #
    #             #print(Qmax_int)
    #             #
    #             def multipole_density_0(r,FF=FF):
    #                 rho_int=quad(lambda q: (q**2)*FF(q*hc)*spherical_jn(L,r*q),self.qrange[0]/hc,Qmax_int/hc,limit=1000) 
    #                 #print(rho_int[0],rho_int[1])
    #                 return 4*pi*rho_int[0]/(2*pi)**3
    #             # vectorize
    #             multipole_density_vec = np.vectorize(multipole_density_0)
    #             # spline
    #             multipole_density_spl = spline_field(multipole_density_vec,"density_"+key0,self.name,rrange=self.rrange,renew=self.renew)
    #             # exponential decay for rho
    #             self.charge_density_dict[key0] = highenergycont_rho(multipole_density_spl,R=self.rrange[1],val=0,t=0)
                
    #             if key0=='M0p':# and (self.charge_density_Mp is None): # -> nasattr
    #                 self.charge_density_Mp = copy.copy(self.charge_density_dict[key0])
                
    #             if key0=='M0n':# and (self.charge_density_Mn is None): # -> hasattr
    #                 self.charge_density_Mn = copy.copy(self.charge_density_dict[key0])
        
    
    def set_scalars_from_rho(self):
        if not hasattr(self,"total_charge"):
            self.set_total_charge()
        if (not hasattr(self,"charge_radius")) or (not hasattr(self,"charge_radius_sq")):
            self.set_charge_radius()
        if (self.k_barrett is not None) and (self.alpha_barrett is not None):
            if not hasattr(self,"barrett_moment"):
                self.set_barrett_moment()
        if hasattr(self,'rhoM0p'):
            if (not hasattr(self,'proton_radius')) or (not hasattr(self,'proton_radius_sq')):
                self.set_proton_radius()
        if hasattr(self,'rhoM0n'):
            if (not hasattr(self,'neutron_radius')) or (not hasattr(self,'neutron_radius_sq')):
                self.set_neutron_radius()
        if hasattr(self,'weak_density'):
            if (not hasattr(self,'weak_radius')) or (not hasattr(self,'weak_radius_sq')):
                self.set_weak_radius()
    
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
        
        if not hasattr(self,"electric field"):
            self.set_electric_field_from_charge_density()

        if not hasattr(self,"electric_potential"):
            self.set_electric_potential_from_electric_field()
        
        if not hasattr(self,"form_factor"):
            self.set_form_factor_from_charge_density()
        
        self.update_dependencies()

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
    fct_q = highenergycutoff_field(fct_q_spl,qrange[1],val=0) # Asymptotic: cutoff to 0
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
    # TODO Adjust r_crit=rrange[1] if rho<0,drho>0 ? beyond oscillatory 
    # highenergy exponential decay for rho
    fct_r = highenergycont_rho(fct_r_spl,rrange[1],val=0,t=0)  # Asymptotic: exp(-r)
    #fct_r = highenergycutoff_field(fct_r_spl,rrange[1],val=0) # alternative
    #
    return fct_r

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
    field_spl=calcandspline(field, rrange, "./test/"+fieldtype+"_"+name+".txt",dtype=float,renew=renew) # <- change path and file name TODO
    return field_spl

def highenergycont_field(field_spl,R,n):
    def field_ultimate(r,R1=R):
        E_crit=field_spl(R1)
        dE=deriv(field_spl,1e-6)
        dE_crit=dE(R1)
        field=highenergycontinuation_poly(r,R1,E_crit,dE_crit,0,n=n)
        if np.any(r<=R1):
            field = field_spl(r)
        if np.size(field)>1:
            field[np.where(r>R1)]=highenergycontinuation_poly(r[np.where(r>R1)],R1,E_crit,dE_crit,0,n=n)
        return field
    return field_ultimate

def highenergycont_rho(field_spl,R,val,t): # often val=0, t=0
    def field_ultimate(r,R1=R):
        E_crit=field_spl(R1)
        dE=deriv(field_spl,1e-6)
        dE_crit=dE(R1)
        field=highenergycontinuation_exp(r,R1,E_crit,dE_crit,val,t=t)
        if np.any(r<=R1):
            field = field_spl(r)
        if np.size(field)>1:
            field[np.where(r>R1)]=highenergycontinuation_exp(r[np.where(r>R1)],R1,E_crit,dE_crit,val,t=t)
        return field
    return field_ultimate

def highenergycutoff_field(field_spl,R,val=np.nan):
    # For r>R return val (default:nan, common choice: 0)
    def field_ultimate(r,R1=R):
        field=np.full(len(np.atleast_1d(r)), val)
        if np.any(r<=R1):
            field = field_spl(r)
        if np.size(field)>1:
            field[np.where(r>R1)]=r[np.where(r>R1)]*val
        return field
    return field_ultimate
