import numpy as np
pi = np.pi

from scipy.linalg import inv

class minimization_measures():
    
    def __init__(self,test_function,x_data,y_data,cov_stat_data,cov_syst_data):
        ''' 
        test_function of type fct(x,*param,**params) 
        x_data and y_data are 1d arrays of the same length 
        cov_stat_data and cov_syst_data are corresponding 2d arrays 
        '''
        
        self.test_function = test_function
        
        self.x_data = np.atleast_1d(x_data)
        self.N_data = len(self.x_data)
        
        self.y_data = np.atleast_1d(y_data)
        
        self.cov_stat_data = np.atleast_2d(cov_stat_data)
        self.cov_syst_data = np.atleast_2d(cov_syst_data)
        
        # check for non-zero off diagonal elements
        self.off_diagonal_covariance = np.sum(np.abs(self.cov_stat_data + self.cov_syst_data)) > np.sum(np.abs(self.cov_stat_data.diagonal() + self.cov_syst_data.diagonal())) 
        
    def set_cov(self,*params_args,**params_kwds):
        
        if np.any(self.cov_syst_data != 0):
            y_test = self.test_function_eval(*params_args,**params_kwds)       
            scale_syst = y_test/self.y_data
            cov_syst_data_rescaled =np.einsum('i,ij,j->ij',scale_syst,self.cov_syst_data,scale_syst)
        else:
            cov_syst_data_rescaled = self.cov_syst_data
        self.cov_data = self.cov_stat_data + cov_syst_data_rescaled
        self.dy_data = np.sqrt(self.cov_data.diagonal()) 
        self.inv_cov_data = inv(self.cov_data) 
        
    def residual(self,*params_args,weighted=True,**params_kwds):
        y_test = self.test_function_eval(*params_args,**params_kwds)
        self.y_test_last_eval = y_test
        return (y_test - self.y_data)/(self.dy_data if weighted else 1. ) 
        
    def loss(self,*params_args,**params_kwds):
        residual = self.residual(*params_args,weighted=False,**params_kwds)
        return np.einsum('i,ij,j',residual,self.inv_cov_data,residual)
    
    def test_function_eval(self,*params_args,**params_kwds):
        return self.test_function(self.x_data,*params_args,**params_kwds)
