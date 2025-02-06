import phasr as phr
import numpy as np

def test_nucleus_FB():
    A_Al27=np.array([0.43418e-1,0.60298e-1,0.28950e-2,-0.23522e-1,-0.79791e-2,0.23010e-2,0.10794e-2,0.12574e-3,-0.13021e-3,0.56563e-4,-0.18011e-4,0.42869e-5])
    R_Al27=7
    nucleus_Al27 = phr.nucleus(name='Al27',Z=13,A=27,ai=A_Al27,R=R_Al27)
    #
    r_test = np.arange(0,7,0.1)
    #
    rho_test = nucleus_Al27.charge_density(r_test)
    rho_test_ref = np.array([7.85286689e-02, 7.85714460e-02, 7.86980954e-02, 7.89035611e-02,
                         7.91793853e-02, 7.95136793e-02, 7.98911133e-02, 8.02929535e-02,
                         8.06971761e-02, 8.10786850e-02, 8.14096597e-02, 8.16600482e-02,
                         8.17982143e-02, 8.17917346e-02, 8.16083298e-02, 8.12169051e-02,
                         8.05886605e-02, 7.96982244e-02, 7.85247584e-02, 7.70529738e-02,
                         7.52740050e-02, 7.31860834e-02, 7.07949670e-02, 6.81140886e-02,
                         6.51644020e-02, 6.19739201e-02, 5.85769566e-02, 5.50131020e-02,
                         5.13259771e-02, 4.75618239e-02, 4.37680013e-02, 3.99914591e-02,
                         3.62772646e-02, 3.26672508e-02, 2.91988487e-02, 2.59041538e-02,
                         2.28092592e-02, 1.99338775e-02, 1.72912492e-02, 1.48883248e-02,
                         1.27261884e-02, 1.08006808e-02, 9.10316986e-03, 7.62141011e-03,
                         6.34043617e-03, 5.24343482e-03, 4.31255180e-03, 3.52959879e-03,
                         2.87664011e-03, 2.33645129e-03, 1.89285427e-03, 1.53094307e-03,
                         1.23721978e-03, 9.99662419e-04, 8.07744512e-04, 6.52421212e-04,
                         5.26090890e-04, 4.22534726e-04, 3.36832199e-04, 2.65248183e-04,
                         2.05088174e-04, 1.54521710e-04, 1.12379512e-04, 7.79358524e-05,
                         5.06926929e-05, 3.01848906e-05, 1.58252972e-05, 6.80469311e-06,
                         2.05464070e-06, 2.72644376e-07])
    #
    np.testing.assert_almost_equal(rho_test/rho_test_ref,1.,decimal=6)
    #
    r_ch_test = nucleus_Al27.charge_radius
    r_ch_test_ref=3.03519934846053
    #
    assert np.abs(r_ch_test - r_ch_test_ref) < 1e-6, f'charge radius should be 3.03519934846053, but is {r_ch_test}'
    #
    Vmin_test = nucleus_Al27.Vmin
    Vmin_test_ref=-0.03955479035793174
    #
    assert np.abs(Vmin_test - Vmin_test_ref) < 1e-6, f'Vmin should be -0.03955479035793174, but is {Vmin_test}'