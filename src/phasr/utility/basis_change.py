def Isospin_basis_to_nucleon_basis(F0,F1,nuc:str):
    if nuc=='p':
        pm=+1
    elif nuc=='n':
        pm=-1
    else:
        raise ValueError("Needs nuc='p','n'")
    return (F0 + pm*F1)/2

def Nucleon_basis_to_isospin_basis(Fp,Fn,iso:str):
    if iso=='0':
        pm=+1
    elif iso=='1':
        pm=-1
    else:
        raise ValueError("Needs I=0,1")
    return Fp + pm*Fn
