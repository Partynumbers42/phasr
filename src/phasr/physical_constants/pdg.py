import os
import numpy as np

path = os.path.join(os.path.dirname(__file__), 'data/mass_width_2021.mcd')

with open( path, "rb" ) as file:
    PDG0 = np.genfromtxt( file,comments=None,skip_header=38,delimiter=[34,18,9,10,18,9,9,21],dtype=['U29',float,float,float,float,float,float,'U21'],autostrip=True,names=['ParticleID', 'Mass', 'posMassErrors', 'negMassErrors', 'Width', 'WidthErrors', 'negWidthErrors', 'Name', 'Charge'])

PDG=np.empty_like(PDG0,dtype=[('ParticleID','U29'),('Mass',float),('posMassErrors',float),('negMassErrors',float),('Width',float),('WidthErrors',float),('negWidthErrors',float),('Name','U21'),('Charge','U21')])
PDG[['ParticleID', 'Mass', 'posMassErrors', 'negMassErrors', 'Width', 'WidthErrors', 'negWidthErrors']]=PDG0[['ParticleID', 'Mass', 'posMassErrors', 'negMassErrors', 'Width', 'WidthErrors', 'negWidthErrors']]
for i in range(len(PDG0)):
    name_inPDG=(PDG0['Name'])[i]
    namecharge_inPDG = str.split(name_inPDG)
    name_inPDG = namecharge_inPDG[0]
    charge_inPDG = namecharge_inPDG[-1]
    if charge_inPDG==name_inPDG:
        charge_inPDG='?'
    PDG['Name'][i]=name_inPDG
    PDG['Charge'][i]=charge_inPDG

def massof(name,charge_str=None,MeV=False,verbose=False): 
    """e.g.: name='pi'"""
    if charge_str!=None:
        if verbose:
            print("looking up", name, "with charge", charge_str,"in MeV" if MeV else "in GeV")
        out = PDG['Mass'][np.argwhere(np.logical_and(PDG['Name']==name, PDG['Charge']==charge_str))]
    else:
        if verbose:
            print("looking up", name,"in MeV" if MeV else "in GeV")
        out = PDG['Mass'][np.argwhere(PDG['Name']==name)]
    
    if MeV==True:
        out*=1e3
    
    if len(out)>1:
        print("More then one particle found.")
    elif len(out)==1:
        out = out[0][0]
    else:
        print("No particle with that signature found.")
    
    return out

def widthof(name,charge_str=None,MeV=False,verbose=False): 
    """e.g.: name='pi'"""
    if charge_str!=None:
        if verbose:
            print("looking up", name, "with charge", charge_str,"in MeV" if MeV else "in GeV")
        out = PDG['Width'][np.argwhere(np.logical_and(PDG['Name']==name, PDG['Charge']==charge_str))]
    else:
        if verbose:
            print("looking up", name,"in MeV" if MeV else "in GeV")
        out = PDG['Width'][np.argwhere(PDG['Name']==name)]
    
    if MeV==True:
        out*=1e3
    
    if len(out)>1:
        print("More then one particle found.")
    elif len(out)==1:
        out = out[0][0]
    else:
        print("No particle with that signature found.")
    
    return out