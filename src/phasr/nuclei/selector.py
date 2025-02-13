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

import os, glob
import numpy as np

def load_reference_nucleus(Z,A,nucleus_key):
    
    reference_data_path = os.path.join(os.path.dirname(__file__), 'data/')
    #print(reference_data_path+"*"+nucleus_key+".txt")
    paths_reference = glob.glob(reference_data_path+"*"+nucleus_key+".txt")
    #print(paths_reference)
    #add loading your own results
    #paths_own = glob.glob(reference_data_path+"*.txt")

    ai_str_list = ['a'+str(i) for i in range(1,17+1,1)]

    references=None

    for path in paths_reference:
        
        with open( path, "rb" ) as file:
            reference_file = np.genfromtxt( file,comments=None,skip_header=2,delimiter=None,dtype=['U5',int,int]+17*[float]+[float],autostrip=True,names=['name', 'A', 'Z']+ai_str_list+['R'])
    	
        references=reference_file[np.argwhere(np.logical_and(reference_file['Z']==Z, reference_file['A']==A))]

    if references is None:
        raise KeyError('No parameterisation on file for this isotope')

    nuclei_ref = []

    for reference in references:
        
        if nucleus_key == 'FB':
            name = reference['name'][0]
            radius = reference['R'][0]
            ai = np.trim_zeros(np.array(list(reference[ai_str_list][0])),trim='b')
            
            nucleus_ref = nucleus(name+'_ref',Z,A,ai=ai,R=radius)
            nuclei_ref.append(nucleus_ref)
   
    if len(nuclei_ref)==1:
        nuclei_ref = nuclei_ref[0]
    
    return nuclei_ref

# TODO
def save_reference_nucleus(Z,A,nucleus_type):
    pass
