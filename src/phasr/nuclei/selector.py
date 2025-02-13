#from .base import nucleus_base
from .parameterizations.fourier_bessel import nucleus_FB
from .parameterizations.oszillator_basis import nucleus_osz
from .parameterizations.fermi import nucleus_fermi
from .parameterizations.basic import nucleus_gauss, nucleus_uniform
from .parameterizations.numerical import nucleus_num
from .parameterizations.coulomb import nucleus_coulomb

def nucleus(name,Z,A,**args):
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
