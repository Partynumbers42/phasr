from .base import nucleus_base
from .parameterisations.fourier_bessel import nucleus_FB
from .parameterisations.oszillator_basis import nucleus_osz
from .parameterisations.numerical import nucleus_num
from .parameterisations.coulomb import nucleus_coulomb

def nucleus(name,Z,A,**args):
    args = {"name":name,"Z":Z,"A":A,**args}
    if ('ai' in args) and ('R' in args):
        return nucleus_FB(**args)
    elif ('Cs' in args):
        return nucleus_osz(**args)
    elif ('charge_density' in args) or  ('electric_field' in args) or  ('electric_potential' in args) or ('formfactor' in args):
        return nucleus_num(**args)
    else:
        return nucleus_coulomb(**args)

# class nucleus(nucleus_base):
#     def __init__(self,name,Z,A,**args):
#         args = {"name":name,"Z":Z,"A":A,**args}
#         if ('ai' in args) and ('R' in args):
#             nucleus_FB.__init__(self,**args)
#         elif ('Cs' in args):
#             nucleus_osz.__init__(self,**args)
#         elif ('charge_density' in args) or  ('electric_field' in args) or  ('electric_potential' in args) or ('formfactor' in args):
#             nucleus_num.__init__(self,**args)
#         else:
#             nucleus_coulomb.__init__(self,**args)
