from . import nucleus

import os #, glob
import numpy as np

reference_data_path = os.path.join(os.path.dirname(__file__), 'data/params_FB.txt')
ai_str_list = ['a'+str(i) for i in range(1,17+1,1)]

with open( reference_data_path, "rb" ) as file:
    reference_file = np.genfromtxt( file,comments=None,skip_header=2,delimiter=None,dtype=['U5',int,int]+17*[float]+[float],autostrip=True,names=['name', 'A', 'Z']+ai_str_list+['R'])

onfile = reference_file[['name','A','Z']]

def load_reference_nucleus(Z,A):
    
    references=reference_file[np.argwhere(np.logical_and(reference_file['Z']==Z, reference_file['A']==A))]

    if len(references)==0:
        raise KeyError('No parameterisation on file for this isotope')

    nuclei_ref = []
    
    counter=1
    for reference in references:
        
        name = reference['name'][0]
        radius = reference['R'][0]
        ai = np.trim_zeros(np.array(list(reference[ai_str_list][0])),trim='b')
        nucleus_ref=nucleus(name+'_ref'+str(counter),Z,A,ai=ai,R=radius)
        nuclei_ref.append(nucleus_ref)
        counter+=1

    if len(nuclei_ref)==1:
        nuclei_ref = nuclei_ref[0]
    
    return nuclei_ref

# TODO
# def save_nucleus(Z,A):
#     pass

# def load_nucleus(Z,A):
#     pass

#reference_data_path = os.path.join(os.path.dirname(__file__), 'data/')
#paths_reference = glob.glob(reference_data_path+"*"+nucleus_key+".txt")
#add loading your own results
#paths_own = glob.glob(reference_data_path+"*.txt")