import numpy as np
pi=np.pi

def angle_shift_mod_pi(phi,n=1): 
    # shift angle to (-pi*n,+pi*n]
    return -((n*pi-phi)%(2*n*pi))+n*pi

def derivative(f,precision=1e-6):
    # fine tuned for numerical stability
    def df(x):
        h=np.abs(x)*precision+precision
        return (f(x+h)-f(x))/h
    return df

def radial_laplace(fct,precision_atzero=1e-3,precision_derivative=1e-6):
    # fine tuned for numerical stability
    dfct = derivative(fct,precision_derivative)
    def r2dfct(r): return r**2 *dfct(r)
    dr2dfct = derivative(r2dfct,precision_derivative)
    def laplacefct(r):
        r_arr = np.atleast_1d(r)
        laplace = np.zeros(len(r_arr))
        mask_r = np.abs(r_arr) > precision_atzero
        if np.any(mask_r):
            laplace[mask_r] = 1/r_arr[mask_r]**2 *dr2dfct(r_arr[mask_r])
        if np.any(~mask_r):
            laplace[~mask_r] = 1/(r_arr[~mask_r]+precision_atzero)**2 *dr2dfct(r_arr[~mask_r]+precision_atzero)
        if np.isscalar(r):
            laplace = laplace[0]
        return laplace
    return laplacefct

from mpmath import hyper, workdps #confluent hypergeometric function
# alternative: fp.hyper
def hyper1f1_scalar_arbitrary_precision(a,b,z,dps=15):
    with workdps(dps):
        return complex(hyper([a],[b],z))
hyper1f1_vector_z_arbitrary_precision=np.vectorize(hyper1f1_scalar_arbitrary_precision,excluded=[0,1,3])
hyper1f1_vector_arbitrary_precision=np.vectorize(hyper1f1_scalar_arbitrary_precision,excluded=[3])

def optimise_radius_highenergy_continuation(fct,x_crit,x_step,x_min=0,fct_limit=0):
    x_crit_initial=x_crit
    fct_crit=fct(x_crit)
    dfct_crit=derivative(fct,1e-6)(x_crit)
    while np.sign(dfct_crit)*np.sign(fct_crit-fct_limit)>0 and x_crit>x_min:
        x_crit=np.max([x_crit-x_step,x_min])
        fct_crit=fct(x_crit)
        dfct_crit=derivative(fct,1e-6)(x_crit)
    if np.sign(dfct_crit)*np.sign(fct_crit-fct_limit)<0:
        if x_crit!=x_crit_initial:
            print("Warning: x_crit adjusted to "+str(x_crit)+", s.t. high-energy continuation is possible")
        return x_crit
    else:
        print("Warning: Did not find a suitable x_crit in ["+str(x_min)+","+str(x_crit_initial)+"]")
        return x_crit_initial

def short_uncertainty_notation(a,das,digits=2):
    
    da=np.max(das)
    digs=(digits-1)-int(np.floor(np.log10(np.abs(da))))
    
    a_out=np.around(a,digs)
    
    das_out_str=[]
    for dai in das:
        dai_out=int(np.around(dai,digs)*10**digs)
        dai_out_str="{da:d}".format(da=dai_out)
        das_out_str.append(dai_out_str)
    
    a_out_str="{a:.{digs}f}".format(a=a_out,digs=digs)
    
    return a_out_str, das_out_str

# energy momentum relations
def energy(momentum,mass):
    return np.sqrt(momentum**2+mass**2)
def momentum(energy,mass):
    return np.sqrt(energy**2-mass**2) if energy > mass else np.sqrt(energy**2-mass**2+0j)

def momentum_transfer(incoming_energy,scattering_angle,mass=np.inf):
    outgoing_energy=incoming_energy*(1-2*incoming_energy*(np.sin(scattering_angle/2)**2)/mass)
    return 2*np.sqrt(incoming_energy*outgoing_energy)*np.sin(scattering_angle/2)