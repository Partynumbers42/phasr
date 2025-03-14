#from .base import nucleus_base
from .parameterizations.fourier_bessel import nucleus_FB
from .parameterizations.oszillator_basis import nucleus_osz
from .parameterizations.fermi import nucleus_fermi
from .parameterizations.basic import nucleus_gauss, nucleus_uniform
from .parameterizations.numerical import nucleus_num
from .parameterizations.coulomb import nucleus_coulomb

def nucleus(name,Z,A,**args):
    '''Select and construct a nucleus object
    
    Keyword arguments:
    name (str) --  internal name, used in temporary file names (should be unique) 
    Z (int) -- Nuclear charge number / proton number
    A (int) -- Atomic mass number / nucleon number
    m (float) -- nucleus mass (default: deduced from Z,A)
    abundance (float) -- natural nucleus abundance (default: deduced from Z,A)
    spin (float) -- nucleus spin (default: deduced from Z,A)
    parity (float) -- nucleus parity (default: deduced from Z,A)
    k_barrett -- k value used for Barrett Moment (default: does not exist)
    alpha_barrett -- alpha value used for Barrett Moment (default: does not exist)
    **args -- parameters for parameterization (selects nucleus_type)
    
    Nucleus types and corresponding parameters:
    'coulomb': None
    'fourier-bessel': ai (1d-array; fm^-3), R (float; fm)
    'oszillator-basis': Ci_dict (dict of 1d-arrays)
    'fermi': c (float; fm), z (float; fm) ,w (float; default=0) 
    'gauss': b (float; fm)
    'uniform': rc (float; fm)
    numerical: charge_density, electric_field electric_potential, form_factor, form_factor_dict, or density_dict (callable / dict of callables)
    '''
    args = {"name":name,"Z":Z,"A":A,**args}
    if ('ai' in args) and ('R' in args):
        return nucleus_FB(**args)
    elif ('Ci_dict' in args):
        return nucleus_osz(**args)
    elif ('c' in args) and ('z' in args):
        return nucleus_fermi(**args)
    elif ('b' in args):
        return nucleus_gauss(**args)
    elif ('rc' in args):
        return nucleus_uniform(**args)
    elif ('charge_density' in args) or  ('electric_field' in args) or  ('electric_potential' in args) or ('form_factor' in args) or ('form_factor_dict' in args) or ('density_dict' in args):
        return nucleus_num(**args)
    else:
        return nucleus_coulomb(**args)
