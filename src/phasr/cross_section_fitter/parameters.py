import numpy as np
pi = np.pi

class parameter_set():
    
    def __init__(self,R:float,total_charge:float,ai=None,ai_abs_bound=None,xi=None):
        
        self.R = R 
        self.total_charge = total_charge
        self.ai_abs_bound = ai_abs_bound
        
        if xi is not None:
            self.xi = xi 
            self.N_x = len(self.xi)
            self.N_a = self.N_x + 1
        elif ai is not None:
            self.ai = ai
            self.N_a = len(self.ai)
            self.N_x = self.N_a - 1
        else:
            raise ValueError("Need to supply either ai or xi!") 
        
        self.ni = np.arange(1,self.N_a+1)
        self.qi=np.arange(1,self.N_a+1)*pi/self.R
        
        if ai_abs_bound is None:
            self.ai_abs_bound = ai_abs_bounds_default(self.ni,self.R,self.total_charge)
        else:
            self.ai_abs_bound = ai_abs_bound
        
        if not hasattr(self,"xi"):
            self.update_ai_then_xi(self.ai)
        
        if not hasattr(self,"ai"):
            self.update_xi_then_ai(self.xi)
        
    def update_ai_then_xi(self,ai):
        self.ai=ai
        self.set_ai_tilde_from_ai()
        self.set_xi_from_ai_tilde()
    
    def update_xi_then_ai(self,xi):
        self.xi=xi
        self.set_ai_tilde_from_xi()
        self.set_ai_from_ai_tilde()
        
    def set_xi_from_ai_tilde(self):
        xi_dummy = np.zeros(self.N_x)
        for j in range(1,self.N_x+1):
            aj_tilde_implied_min, aj_tilde_implied_max = ai_tilde_implied_bounds(j,self.ai_tilde,self.ai_abs_bound,self.qi,self.N_a,self.R,self.total_charge)
            xi_dummy[j-1] = (self.ai_tilde[j-1] - aj_tilde_implied_min[:-1])/(aj_tilde_implied_max[:-1]-aj_tilde_implied_min[:-1]) 
        # make sure xi are between 0 and 1
        xi_dummy[xi_dummy<0]=0
        xi_dummy[xi_dummy>1]=1
        
        self.xi = xi_dummy
    
    def set_ai_tilde_from_xi(self):
        ai_tilde_dummy = np.zeros(self.N_a)
        for j in range(1,self.N_a+1):
            aj_tilde_implied_min, aj_tilde_implied_max = ai_tilde_implied_bounds(j,self.ai_tilde,self.ai_abs_bound,self.qi,self.N_a,self.R,self.total_charge)
            ai_tilde_dummy[j-1] = self.xi[j-1]*(aj_tilde_implied_max-aj_tilde_implied_min) + aj_tilde_implied_min if j<self.N_a else aj_tilde_implied_min
        self.ai_tilde = ai_tilde_dummy
    
    def set_ai_tilde_from_ai(self):
        self.ai_tilde = self.ai*(-1)**(self.ni+1)
    
    def set_ai_from_ai_tilde(self):
        self.ai = self.ai_tilde*(-1)**(self.ni+1)
    
    def get_ai(self):
        return self.ai
    
    def get_xi(self):
        return self.xi

def ai_tilde_implied_bounds(j,ai_tilde,ai_abs_bound,qi,N_a,R,total_charge):
    
    qj = qi[j-1]
    mean = total_charge/(4*pi*R) - np.sum(ai_tilde[0:j-1]/qi[0:j-1]**2)
    spread = np.sum(ai_abs_bound[j:N_a+1]/qi[j:N_a+1]**2)

    ai_tilde_implied_max=min(qj**2*(mean + spread),ai_abs_bound[j-1])
    ai_tilde_implied_min=max(qj**2*(mean - spread),-ai_abs_bound[j-1])

    return  ai_tilde_implied_min, ai_tilde_implied_max

def ai_abs_bounds_default(ni,R,Z):
    return (1./ni)*(Z*pi**2)/(2*R**3)