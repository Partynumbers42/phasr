import numpy as np # type: ignore

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