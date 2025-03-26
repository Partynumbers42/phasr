import numpy as np

def Tslope(t,val_crit,deriv_crit,limit):
    """T(t) ~ e^(-t)"""
    return ((limit-val_crit)/deriv_crit)*np.exp(-t)

def highenergy_continuation_exp(s,scrit,val_crit,deriv_crit,limit,t=0):
    """ goes like e^(-s/T(t)) """
    T = Tslope(t,val_crit,deriv_crit,limit)
    phase = limit + (val_crit-limit + (deriv_crit + (val_crit-limit)/Tslope(t,val_crit,deriv_crit,limit))*(s-scrit))*np.exp(-(s-scrit)/T)
    return phase

def highenergy_continuation_poly(s,scrit,val_crit,limit,n=1):
    """ goes like 1/s^n """
    return limit - scrit**n * (limit-val_crit)/s**n

def highenergy_continuation_poly_offset(s,scrit,val_crit,deriv_crit,limit,n=1):
    """ goes like 1/(a + b s)^n """
    return limit + (limit-val_crit)/((scrit-((s**n)/(scrit**(n-1))))*(deriv_crit/(n*(limit-val_crit))) - 1)

def highenergy_continuation_log(s,scrit,val_crit,deriv_crit):
    """ goes/diverges like log(s) """
    return val_crit + deriv_crit*scrit*np.log(s/scrit)

def highenergy_continuation_trig(s,scrit,val_crit,deriv_crit,q):
    """ goes/oszillates like A*sin(q*s + phi) """
    s0 = np.arctan2(val_crit*q,deriv_crit)/q - scrit
    A = val_crit/np.sin(q*(scrit+s0))
    return A*np.sin(q*(s+s0))