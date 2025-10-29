#!/usr/bin/env python
# coding: utf-8

from .pdg import massof
from .iaea_nds import massofnucleus
import numpy as np

class constants():
    # physical constants
    alpha_el=1./137.035999084
    hc=197.3269804 # MeV fm
    fermi_constant= 1.1663788e-11 #(6) MeV^-2
    sin_sq_theta_weinberg = 0.23129 #(4) PDG value at M_Z in MSbar
    W_mass_over_Z_mass = 0.88136 #(15) = m_W / m_Z
    
    # proton/neutron charge radii and anomalous magnetic moment
    # from Hoferichter 2020
    rsq_p=0.7071#(7) #fm^2
    rsq_n=-0.1161#(22) #fm^2
    kappa_p=1.79284734462#(82)
    kappa_n=-1.91304273#(45)
    rsq_sN=-0.0048#(6) #fm^2
    kappa_sN=-0.017#(4) #fm^2

    # weak charge per proton/neutron
    Qw_p=0.0710#(4) old:0.0714
    Qw_n=-0.9891#(3) old:-0.9900
    
    # decay constants
    fpi=130.2/np.sqrt(2.)
    fK=155.7/np.sqrt(2.)


# masses
class masses():
    # leptons
    mtau=massof('tau',MeV=True)
    mmu=massof('mu',MeV=True)
    me=massof('e',MeV=True)
    # mesons
    mpi0 = massof('pi','0',MeV=True)
    mpipl = massof('pi','+',MeV=True)
    mK0 = massof('K','0',MeV=True)
    mKpl = massof('K','+',MeV=True)
    meta = massof("eta",MeV=True)
    metap = massof("eta'(958)",MeV=True)
    # baryons
    mp=massof(name='p',MeV=True)
    mn=massof(name='n',MeV=True)
    mN=(mn+mp)/2
    # masses nuclei
    mAl27=massofnucleus('Al',27)
    mCa40=massofnucleus('Ca',40)
    mCa48=massofnucleus('Ca',48)
    mTi46=massofnucleus('Ti',46)
    mTi48=massofnucleus('Ti',48)
    mTi50=massofnucleus('Ti',50)
    mAu197=massofnucleus('Au',197)
    mPb208=massofnucleus('Pb',208)

# unit_trafo 
class trafos():
    #
    hc = constants.hc
    alpha_el = constants.alpha_el
    mmu = masses.mmu
    #
    cmsq_to_mub = 1e30
    mub_to_fmsq = 1e-4
    cmsq_to_fmsq = cmsq_to_mub*mub_to_fmsq
    mb_to_fmsq = 1e3*mub_to_fmsq
    invMeVsq_to_fmsq = hc**2
    