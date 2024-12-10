from .. import constants
from ..utility import calcandspline
from ..utility.continuer import highenergycontinuation_exp, highenergycontinuation_poly

import numpy as np  # type: ignore
pi = np.pi

from scipy.integrate import quad # type: ignore

from scipy.special import spherical_jn # type: ignore

from scipy.misc import derivative as deriv # type: ignore <-  TODO: update to own method 

def range_seperator(xrange,fct):
    Xmin_int=xrange[0]
    if fct(xrange[1]+xrange[2])==0:
        Xmax_int=xrange[1]
        return [Xmin_int, Xmax_int]
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
        integrali = quad(integrand,Rmin,Rmax,limit=1e3)[0]
        integral += integrali 
    return integral

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
        radius = np.sqrt(radius_sq)
    return radius_sq, radius

def spline_field(field,fieldtype,name,rrange,renew):
    # spline
    field_spl=calcandspline(field, rrange, "./lib/splines/"+fieldtype+"_"+name+".txt",dtype=float,renew=renew)
    return field_spl

def highenergycont_field(field_spl,R,n):
    def field_ultimate(r,R1=R):
        E_crit=field_spl(R1)
        dE_crit=deriv(field_spl,R1,1e-6)
        field=highenergycontinuation_poly(r,R1,E_crit,dE_crit,0,n=n)
        if np.any(r<=R1):
            field = field_spl(r)
        if np.size(field)>1:
            field[np.where(r>R1)]=highenergycontinuation_poly(r[np.where(r>R1)],R1,E_crit,dE_crit,0,n=n)
        return field
    return field_ultimate
#
def highenergycont_rho(field_spl,R,val,t): # often val=0, t=0
    def field_ultimate(r,R1=R):
        E_crit=field_spl(R1)
        dE_crit=deriv(field_spl,R1,1e-6)
        field=highenergycontinuation_exp(r,R1,E_crit,dE_crit,val,t=t)
        if np.any(r<=R1):
            field = field_spl(r)
        if np.size(field)>1:
            field[np.where(r>R1)]=highenergycontinuation_exp(r[np.where(r>R1)],R1,E_crit,dE_crit,val,t=t)
        return field
    return field_ultimate
#
def highenergycutoff_field(field_spl,R,val=np.nan):
    # For r>R return val (default:nan)
    def field_ultimate(r,R1=R):
        field=r*val
        if np.any(r<=R1):
            field = field_spl(r)
        if np.size(field)>1:
            field[np.where(r>R1)]=r[np.where(r>R1)]*val
        return field
    return field_ultimate

class nucleus_base:
    def __init__(self,name,Z, A, m=None, abundance=None, spin=None, parity=None, Qw=None, 
                 rrange=[0.,50.,0.05], qrange=[0.,2000.,2.], 
                 pickleable=False, calc=None, renew=False, save=True,
                 #barebone=False, pickleable=False, renew=False, calc=None, calc_multipoles=None, 
                 #spline_hyp1f1=None, fp=False, ap_dps=15, 
                 **args):
        #
        self.nucleus_type = "base"
        self.name = name
        self.Z = Z
        self.A = A
        #
        self.m = m
        if self.m is None:
            self.lookup_nucleus_mass()
        self.abundance=abundance
        if self.abundance is None:
            self.lookup_nucleus_abundance()
        self.spin=spin
        self.parity=parity
        if (self.spin is None) or (self.parity is None):
            self.lookup_nucleus_JP()
        #
        Qw_p=constants.Qw_p
        Qw_n=constants.Qw_n
        #
        self.weak_charge = Qw
        if (self.weak_charge is None):
            self.weak_charge = self.Z*Qw_p + (self.A-self.Z)*Qw_n
        #
        self.rrange=rrange #fm
        self.qrange=qrange #MeV
        #
        self.pickleable=pickleable # no structures, only simple data types 
        self.renew=renew # overwrite existing calculations 
        self.save=save
        self.calc=calc
        #
        #self.barebone=barebone # only params, Q, V(r), V(0) 
        #if self.renew:
        #    self.calc=True
        #self.calc_multipoles = calc_multipoles
        #if self.calc_multipoles is None:
        #    self.calc_multipoles=['M0p','M0n']
        #
        # initialize attributes
        self.total_charge = None
        self.weak_charge = None
        self.charge_radius = None
        self.charge_density = None
        self.electric_field = None
        self.electric_potential = None
        self.formfactor = None
        self.Vmin = None
        #
    def lookup_nucleus_mass(self):
        self.m = None #pdg.massofnucleusZN(self.Z,self.A-self.Z) <---- TODO
    
    def lookup_nucleus_abundance(self):
        self.abundance = None #pdg.abundanceofnucleusZN(self.Z,self.A-self.Z) <---- TODO
            
    def lookup_nucleus_JP(self):
        JP = None#pdg.JPofnucleusZN(self.Z,self.A-self.Z)
        if type(JP) is tuple:
            J , P = JP
            if self.spin is not None and J!=self.spin:
                raise ValueError('looked up spin J='+str(J)+' different to present one J='+str(self.spin))
            if self.parity is not None and P!=self.parity:
                raise ValueError('looked up parity P='+str(P)+' different to present one P='+str(self.parity))
            self.spin, self.parity = J, P
    
    def wanna_calc(self,quantity_name='Some quantity'):
       if self.calc is None:
           yn=input(quantity_name+" of "+self.name+" is not available analytically, do you want to calculate it (you won't be asked again for others)? (y/n): ")
           if yn=='n' or yn=='no' or yn=='No' or yn=='N':
               self.calc=False
           else:
               self.calc=True
    
    def set_total_charge(self):
       #self.wanna_calc('Total charge') # <- move outside TODO
       #if not self.calc:
       #    return None
       self.total_charge=calc_charge(self.charge_density,self.rrange)
        
    def set_charge_radius(self,norm=None):
        #self.wanna_calc('Charge radius')
        #if not self.calc:
        #    return None
        if norm is None:
            norm=self.total_charge
        self.charge_radius_sq, self.charge_radius = calc_radius(self.charge_density,self.rrange,norm)
   
    def set_proton_radius(self,norm=None):
        self.wanna_calc('Proton radius')
        #if not self.calc:
        #    return None
        if norm is None:
            norm=self.Z
        self.proton_radius_sq, self.proton_radius = calc_radius(self.charge_density_Mp,self.rrange,norm)
   
    def set_neutron_radius(self,norm=None):
        self.wanna_calc('Neutron radius')
        #if not self.calc:
        #     return None
        if norm is None:
            norm=self.A-self.Z
        self.neutron_radius_sq, self.neutron_radius = calc_radius(self.charge_density_Mn,self.rrange,norm)
 
    def set_weak_radius(self,norm=None):
        self.wanna_calc('Weak radius')
        #if not self.calc:
        #    return None
        if norm is None:
            norm=self.weak_charge
        self.weak_radius_sq, self.weak_radius = calc_radius(self.charge_density_dict['rhow0'],self.rrange,norm)

    def set_Vmin(self):
        self.wanna_calc('Vmin')
        #if not self.calc:
        #    return None
        if not self.barebone:
            self.Vmin = np.min(self.electric_potential(np.arange(*self.rrange)))
        else:
            self.Vmin = self.electric_potential(self.rrange[0])

    def set_El_from_rho(self):
        #
        #self.wanna_calc('Electric field')
        #if not self.calc:
        #    return None
        #
        def electric_field_0(r,rho=self.charge_density):
            charge_r = quad_seperator(lambda x: (x**2)*rho(x),[0,r])
            return np.sqrt(4*pi*constants.alpha_el)/(r**2)*charge_r if r!=0. else 0.  # as long as rho(0) finite follows E(0)=0, #*e=np.sqrt(4*pi*alpha_el)
        # vectorize
        electric_field_vec = np.vectorize(electric_field_0)
        # spline
        electric_field_spl = spline_field(electric_field_vec,"electric_field",self.name,rrange=self.rrange,renew=self.renew)
        # highenery continue
        self.electric_field = highenergycont_field(electric_field_spl,R=self.rrange[1]*0.95,n=2) # Asymptotic: 1/r^2

    def set_V_from_El(self):
        #
        Rs0 = range_seperator(self.rrange,self.electric_field)
        def electric_potential_0(r,El=self.electric_field):
            Rs=np.array([r,*Rs0[Rs0>r]])
            potential_r = quad_seperator(El,Rs)
            return - np.sqrt(4*pi*constants.alpha_el)*potential_r #*e=np.sqrt(4*pi*alpha_el)
        # vectorize
        electric_potential_vec = np.vectorize(electric_potential_0)
        # spline
        electric_potential_spl = spline_field(electric_potential_vec,"electric_potential",self.name,rrange=self.rrange,renew=self.renew)
        # highenery continue
        self.electric_potential = highenergycont_field(electric_potential_spl,R=self.rrange[1]*0.95*0.95,n=1) # Asymptotic: 1/r

    def set_FF_from_rho(self):
        #
        #self.wanna_calc('Form factor')
        #if not self.calc:
        #    return None
        #
        Rs = range_seperator(self.rrange,self.density)
        #
        def formfactor_0(q,rho=self.charge_density,Z=self.total_charge):
            formfactor_int = quad_seperator(lambda r: (r**2)*rho(r)*spherical_jn(0,q/constants.hc*r),Rs)
            return 4*pi*formfactor_int/Z
        # vectorize
        formfactor_vec = np.vectorize(formfactor_0)
        # spline
        formfactor_spl = spline_field(formfactor_vec,"formfactor",self.name,rrange=self.qrange,renew=self.renew)
        # highenery cut off at qmax
        self.formfactor = highenergycutoff_field(formfactor_spl,R=self.qrange[1],val=0) # Asymptotic: cutoff to 0

    def set_rho_from_FF(self):
        #
        #self.wanna_calc('Charge density')
        #if not self.calc:
        #    return None
        #
        Qs = range_seperator(self.qrange,self.formfactor)
        #
        def charge_density_0(r,FF=self.formfactor,norm=self.total_charge):
            rho_int=quad_seperator(lambda q: (q**2)*FF(q)*spherical_jn(0,r/constants.hc*q)/constants.hc**3,Qs) 
            return 4*pi*rho_int*norm/(2*pi)**3
        # vectorize
        charge_density_vec = np.vectorize(charge_density_0)
        # spline
        charge_density_spl = spline_field(charge_density_vec,"charge_density",self.name,rrange=self.rrange,renew=self.renew)
        # highenery cut off at rmax
        #self.charge_density = highenergycutoff_field(charge_density_spl,R=self.rrange[1],val=0)        
        # exponential decay for rho
        # TODO add way to move r_crit back if rho<0,drho>0 beyond oscillatory        
        self.charge_density = highenergycont_rho(charge_density_spl,R=self.rrange[1],val=0,t=0)  # Asymptotic: exp(-r)
    
    # TODO add these functions as derivatives
    
    def set_rho_from_El(self):
        pass
    
    def set_El_from_V(self):
        pass
    
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
                
    #         if L%2==0 and (key_Fch in self.formfactor_dict):
    #             #
    #             #print('calc even L='+str(L))
    #             #
    #             if L==0 and (self.charge_density is not None):
    #                 print('set L=0 from self.charge density')
    #                 self.charge_density_dict[key_rho]=self.charge_density
    #             else:     
    #                 Fch = self.formfactor_dict[key_Fch]
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
    #                 if L==0 and (self.charge_density is None):
    #                     #print('overwrite L=0 self.charge_density')
    #                     self.charge_density = copy.copy(self.charge_density_dict[key_rho])
    #             #
    #             Fw = self.formfactor_dict[key_Fw]
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
    #         elif L%2==1 and (key_Fmag in self.formfactor_dict):
    #             #
    #             #print('calc odd L='+str(L))
    #             #
    #             Fmag = self.formfactor_dict[key_Fmag]
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
    #             FF = self.formfactor_dict[key_FF+'c'] # only with CMS corrections
                
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
                
    #             if key0=='M0p':# and (self.charge_density_Mp is None):
    #                 self.charge_density_Mp = copy.copy(self.charge_density_dict[key0])
                
    #             if key0=='M0n':# and (self.charge_density_Mn is None):
    #                 self.charge_density_Mn = copy.copy(self.charge_density_dict[key0])
        
    
    def set_scalars_from_rho(self):
        self.set_total_charge()
        self.set_charge_radius()
        # try:
        #     if self.charge_density_Mp is not None:
        #         self.set_proton_radius()
        # except: 
        #     pass
        # try:
        #     if self.charge_density_Mn is not None:
        #         self.set_neutron_radius()
        # except: 
        #     pass
        # try:
        #     if self.charge_density_dict['rhow0'] is not None:
        #         self.set_weak_radius()
        # except: 
        #     pass
            
    def fill_gaps(self):
        
        if self.charge_density is None:
            if self.formfactor is not None:    
                self.set_rho_from_FF()
            else:
                if self.electric_field is None:
                    if self.electric_potential is None:
                        raise ValueError("Not enough information to deduce the charge density")
                    raise ValueError('still need to add this, sorry') #self.set_El_from_V() #TODO
                raise ValueError('still need to add this, sorry') #self.set_rho_from_El() #TODO
                
        if self.electric_potential is None:
            if self.electric_field is None:
                if self.charge_density is None:
                    if self.formfactor is None:
                        raise ValueError("Not enough information to deduce the electric potential")
                    self.set_rho_from_FF()
                self.set_El_from_rho()
            self.set_V_from_El()
        
        if self.formfactor is None:
            if self.charge_density is None:
                if self.electric_field is None:
                    if self.electric_potential is None:
                        raise ValueError("Not enough information to deduce the form factor")
                    raise ValueError('still need to add this, sorry') #self.set_El_from_V() #TODO
                raise ValueError('still need to add this, sorry') #self.set_rho_from_El() #TODO
            self.set_FF_from_rho()

        if self.electric_field is None:
            if self.electric_potential is not None:
                raise ValueError('still need to add this, sorry') #self.set_El_from_V() #TODO
            else:
                if self.charge_density is None:
                    if self.formfactor is not None:
                        raise ValueError("Not enough information to deduce the electric field")
                    self.set_rho_from_FF()
                self.set_El_from_rho()   
        
    # def fill_gaps(self):
    #     while self.electric_potential is None:
    #         while self.electric_field is None:
    #             while self.charge_density is None:
    #                 while self.formfactor is None:
    #                     raise ValueError('no structure (FF,rho,El,V) given to start from')
    #                 self.set_rho_from_FF()
    #             self.set_El_from_rho()
    #         self.set_V_from_El()
        
    #     while self.formfactor is None:
    #         while self.charge_density is None:
    #             while self.electric_field is None:
    #                 while self.electric_potential is None:
    #                     raise ValueError('no structure (FF,rho,El,V) given to start from')
    #                 raise ValueError('still need to add this, sorry') #self.set_El_from_V()
    #             raise ValueError('still need to add this, sorry') #self.set_rho_from_El()
    #         self.set_FF_from_rho()

    #     if self.charge_density is None:
    #         self.set_rho_from_FF()
    #     if self.electric_field is None:
    #         self.set_El_from_rho()