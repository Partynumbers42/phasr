import os
import re
import numpy as np

path = os.path.join(os.path.dirname(__file__), 'data/nuclei_2022.csv')
# https://www-nds.iaea.org/relnsd/vcharthtml/VChartHTML.html

with open( path, "rb" ) as file:
    nuclii = np.genfromtxt( file,comments=None,delimiter=',',names=True,dtype=2*[int]+['<U3']+4*[float]+['<U10']+3*[float]+['<U10']+['<U10','<U3',float]+4*['<U10',float,float]+23*[float]+3*['<U50'])
#
u_to_MeV=931.49410242 
#
def massofnucleusZN(Z,N,eV=True,MeV=True): # in MeV
    if eV:
        if MeV:
            pref=1e-6*u_to_MeV #MeV
        else:
            pref=1e-9*u_to_MeV #GeV
    else:
        pref=1e-6 #u
    mass = pref*nuclii['atomic_mass'][np.argwhere(np.logical_and(nuclii['z']==Z, nuclii['n']==N))][0,0]
    return mass
#
def massofnucleus(name,A,eV=True,MeV=True): # in MeV
    if eV:
        if MeV:
            pref=1e-6*u_to_MeV #MeV
        else:
            pref=1e-9*u_to_MeV #GeV
    else:
        pref=1e-6 #u
    mass = pref*nuclii['atomic_mass'][np.argwhere(np.logical_and(nuclii['symbol']==name,nuclii['z']+nuclii['n']==A))][0,0]
    return mass
#
def abundanceofnucleus(name,A):
    abundance = nuclii['abundance'][np.argwhere(np.logical_and(nuclii['symbol']==name,nuclii['z']+nuclii['n']==A))][0,0]
    return abundance/100
#
def abundanceofnucleusZN(Z,N):
    abundance = nuclii['abundance'][np.argwhere(np.logical_and(nuclii['z']==Z, nuclii['n']==N))][0,0]
    return abundance/100
# 
# last line says stable (1) or unstable (0)
def abundantIsotopes(name):
    isos = nuclii[np.logical_and(nuclii['symbol']==name,nuclii['abundance']==nuclii['abundance'])]
    return np.stack([isos['z']+isos['n'],isos['abundance'],isos['half_life']=='STABLE'],axis=-1)
# 
# be carefull some are non-stable but decay slow enough to still be abundant
def stableIsotopes(name):
    isos = nuclii[np.logical_and(nuclii['symbol']==name,nuclii['half_life']=='STABLE')]
    return np.stack([isos['z']+isos['n'],isos['abundance']],axis=-1)
#

p_half=re.compile(r'\(?([0-9]{1,}/2)\)?\(?([+-])\)?')
p_full=re.compile(r'\(?[^/]?([0-9]{1,})\)?\(?([+-])\)?')
#
def JPofnucleus(name,A):
    JP = nuclii['jp'][np.argwhere(np.logical_and(nuclii['symbol']==name,nuclii['z']+nuclii['n']==A))][0,0]
    j_half=p_half.search(JP)
    if j_half:
        Jo2=j_half.group(1)
        P=j_half.group(2)
        if JP!=Jo2+P:
            print("Warning: JP="+JP+", selected "+Jo2+P)
        return float(Jo2[:-2])/2, int(P+'1')
    j_full=p_full.search(JP)
    if j_full:
        J=j_full.group(1)
        P=j_full.group(2)
        if JP!=J+P:
            print("Warning: JP="+JP+", selected "+J+P)
        return float(J), int(P+'1')
    if JP==' ':
        return np.nan, np.nan
    print('Warning: Could not make sense from JP='+JP,", returning JP as str")
    return JP
#
def JPofnucleusZN(Z,N):
    JP = nuclii['jp'][np.argwhere(np.logical_and(nuclii['z']==Z, nuclii['n']==N))][0,0]
    j_half=p_half.search(JP)
    if j_half:
        Jo2=j_half.group(1)
        P=j_half.group(2)
        if JP!=Jo2+P:
            print("Warning: JP="+JP+", selected "+Jo2+P)
        return float(Jo2[:-2])/2, int(P+'1')
    j_full=p_full.search(JP)
    if j_full:
        J=j_full.group(1)
        P=j_full.group(2)
        if JP!=J+P:
            print("Warning: JP="+JP+", selected "+J+P)
        return float(J), int(P+'1')
    if JP==' ':
        return np.nan, np.nan
    print('Warning: Could not make sense from JP='+JP,", returning JP as str")
    return JP
#
