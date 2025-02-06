import phasr as phr
import numpy as np

def setup_test_nucleus():
    A_Al27=np.array([0.43418e-1,0.60298e-1,0.28950e-2,-0.23522e-1,-0.79791e-2,0.23010e-2,0.10794e-2,0.12574e-3,-0.13021e-3,0.56563e-4,-0.18011e-4,0.42869e-5])
    R_Al27=7
    return phr.nucleus(name='Al27',Z=13,A=27,ai=A_Al27,R=R_Al27)

def test_groundstate_energy():
    test_nucleus = setup_test_nucleus()
    test_boundstates = phr.boundstates(test_nucleus,kappa=-1,lepton_mass=phr.masses.mmu,renew=True,save=False)
    groundstate_energy = test_boundstates.energy_levels[0]
    groundstate_energy_ref = -0.465038380152165
    assert np.abs(groundstate_energy-groundstate_energy_ref)/groundstate_energy_ref < 1e-12, f'groundstate energy should be -0.4590348117169327, but is {groundstate_energy}'

def test_phase_difference():
    test_nucleus = setup_test_nucleus()
    test_continuumstate = phr.continuumstates(test_nucleus,kappa=-1,energy=150,renew=True,save=False)
    test_continuumstate.extract_phase_shift()
    phase_difference = test_continuumstate.phase_difference
    phase_difference_ref = -0.06179085234971993
    assert np.abs(phase_difference - phase_difference_ref)/phase_difference_ref < 1e-12, f'phase difference should be -0.06179085234971993, but is {phase_difference}'
