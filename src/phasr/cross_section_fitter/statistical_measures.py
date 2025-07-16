import numpy as np
pi = np.pi

from scipy.linalg import inv

class minimization_measures():
    
    def __init__(self,test_function,x_data,y_data,cov_stat_data,cov_syst_data):
        ''' 
        test_function of type fct(x,*param,**params) 
        x_data, y_data, dy_data are 1d arrays of the same length
        '''
        
        self.test_function = test_function
        
        self.x_data = x_data
        self.N_data = len(self.x_data)
        
        self.y_data = y_data
        
        self.cov_stat_data = cov_stat_data
        self.cov_syst_data = cov_syst_data
        
        self.set_cov()
    
    def set_cov(self,*params_args,**params_kwds):
        y_test = self.test_function(self.x_data,*params_args,**params_kwds)       
        scale_syst = y_test/self.y_data
        cov_syst_data_rescaled =np.einsum('i,ij,j->ij',scale_syst,self.cov_syst_data,scale_syst)
        self.cov_data = self.cov_stat_data + cov_syst_data_rescaled
        self.inv_cov_data = inv(self.cov_data)
    
    def residual(self,*params_args,**params_kwds):
        y_test = self.test_function(self.x_data,*params_args,**params_kwds)       
        return (y_test - self.y_data)/self.dy_data
    
    def loss(self,*params_args,**params_kwds):
        residual = self.residual(*params_args,**params_kwds)
        return np.einsum('i,ij,j',residual,self.inv_cov_data,residual)


class Parameter_set():
    def __init__(self,R:float,ai=None,xi=None):
        
        self.R = R
        self.xi = xi 
        self.ai = ai
        self.alim = (1./nu)*(Z*pi**2)/(2*R**3)
    
    
    def update_dependencies():
        pass
        
    


def a_lim(nu,Z,R):
    return (1./nu)*(Z*pi**2)/(2*R**3)

def xi_ai(ai,ai_tilde_lim,Z,R):

    N=len(ai_tilde_lim)
    ai_t=ai_tilde(ai)
    nus=np.arange(1,N+1)

    ai_tilde_min, ai_tilde_max = ai_impl_border_vec(nus,ai_t,ai_tilde_lim,Z,R)

    xi=(ai_t[:-1] - ai_tilde_min[:-1])/(ai_tilde_max[:-1]-ai_tilde_min[:-1])

    return xi
