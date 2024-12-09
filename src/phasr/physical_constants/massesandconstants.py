#!/usr/bin/env python
# coding: utf-8

# physical constants
alpha_el=1./137.035999084
hc=197.3269804 # MeV fm
GF= 1.1663788e-11 #(6) MeV^-2

# decay constants
fpi=130.2/np.sqrt(2.)
fK=155.7/np.sqrt(2.)

# masses particles


# masses nuclei



# unit_trafo    
cmsq_to_mub = 1e30
mub_to_fmsq = 1e-4
fmsq_to_invMeVsq = 1./hc**2
invMeVsq_to_invmmualphasq = (mmu*alpha_el)**2
# 
fmsq_to_invmmualphasq = fmsq_to_invMeVsq*invMeVsq_to_invmmualphasq
mub_to_invmmualphasq = mub_to_fmsq*fmsq_to_invmmualphasq
cmsq_to_invmmualphasq = cmsq_to_mub*mub_to_invmmualphasq
# 
invmmualphasq_to_fmsq = 1./fmsq_to_invmmualphasq