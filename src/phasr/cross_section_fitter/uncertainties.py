import numpy as np
pi = np.pi

def add_syst_uncert(best_key,best_fits):
    
    best_fit=best_fits[best_key]

    _, _, pars_u, pars_l = da_syst_gen(best_key,best_fits,omega=10,rrange=[0,10,1e-1],verbose=False)
    
    _,dxi_syst_u,_ = params_to_array(pars_u,key0='R',name='dx',name_lim='a')
    _,dxi_syst_l,_ = params_to_array(pars_l,key0='R',name='dx',name_lim='a')
    
    dxi_syst = np.max([dxi_syst_u,dxi_syst_l],axis=0)
    
    dxi=best_fit['dx']
    xi=best_fit['x']
    alim=best_fit['alim']
    R = best_fit['R']
    Z = best_fit['Z']
    ai = best_fit['a']
    N = best_fit['N_x']
    nu=np.arange(1,N+1+1)
    qi = nu*pi/R
    k, alpha = best_fit['k'], best_fit['alpha']
    #
    redchisq_fit = best_fit['redchisq']
    dxi_model = np.sqrt(redchisq_fit*dxi**2+dxi_syst**2) #dxi_model is already rescaled with chi^2 (the statistical part)
    dxi_model_u = np.sqrt(redchisq_fit*dxi**2+dxi_syst_u**2)
    dxi_model_l = np.sqrt(redchisq_fit*dxi**2+dxi_syst_l**2)
    #
    cov_xi=best_fit['cov_x']
    corr_xi=np.einsum('i,ij,j->ij',1./dxi,cov_xi,1./dxi)
    #
    cov_xi_syst=np.einsum('i,ij,j->ij',dxi_syst,corr_xi,dxi_syst)
    cov_xi_syst_u=np.einsum('i,ij,j->ij',dxi_syst_u,corr_xi,dxi_syst_u)
    cov_xi_syst_l=np.einsum('i,ij,j->ij',dxi_syst_l,corr_xi,dxi_syst_l)
    cov_xi_model=np.einsum('i,ij,j->ij',dxi_model,corr_xi,dxi_model)
    cov_xi_model_u=np.einsum('i,ij,j->ij',dxi_model_u,corr_xi,dxi_model_u)
    cov_xi_model_l=np.einsum('i,ij,j->ij',dxi_model_l,corr_xi,dxi_model_l)
    #
    _,cov_ai_syst=ai_xi(xi,alim,Z,R,Cov=cov_xi_syst)
    _,cov_ai_syst_u=ai_xi(xi,alim,Z,R,Cov=cov_xi_syst_u)
    _,cov_ai_syst_l=ai_xi(xi,alim,Z,R,Cov=cov_xi_syst_l)
    _,cov_ai_model=ai_xi(xi,alim,Z,R,Cov=cov_xi_model)
    _,cov_ai_model_u=ai_xi(xi,alim,Z,R,Cov=cov_xi_model_u)
    _,cov_ai_model_l=ai_xi(xi,alim,Z,R,Cov=cov_xi_model_l)
    #
    dai_syst=np.sqrt(np.diagonal(cov_ai_syst))
    dai_syst_u=np.sqrt(np.diagonal(cov_ai_syst_u))
    dai_syst_l=np.sqrt(np.diagonal(cov_ai_syst_l))
    dai_model=np.sqrt(np.diagonal(cov_ai_model))
    dai_model_u=np.sqrt(np.diagonal(cov_ai_model_u))
    dai_model_l=np.sqrt(np.diagonal(cov_ai_model_l))
    #
    syst_param_uncert_dict={'dx_syst':dxi_syst,'dx_syst_upper':dxi_syst_u,'dx_syst_lower':dxi_syst_l,'dx_model':dxi_model,'dx_model_upper':dxi_model_u,'dx_model_lower':dxi_model_l,'cov_x_syst':cov_xi_syst,'cov_x_syst_upper':cov_xi_syst_u,'cov_x_syst_lower':cov_xi_syst_l,'cov_x_model':cov_xi_model,'cov_x_model_upper':cov_xi_model_u,'cov_x_model_lower':cov_xi_model_l,'da_syst':dai_syst,'da_syst_upper':dai_syst_u,'da_syst_lower':dai_syst_l,'da_model':dai_model,'da_model_upper':dai_model_u,'da_model_lower':dai_model_l,'cov_a_syst':cov_ai_syst,'cov_a_syst_upper':cov_ai_syst_u,'cov_a_syst_lower':cov_ai_syst_l,'cov_a_model':cov_ai_model,'cov_a_model_upper':cov_ai_model_u,'cov_a_model_lower':cov_ai_model_l}
    #
    dr_syst=atom.charge_radius_FB_uncertainty(ai,Z,qi,N+1,R,cov_ai_syst)
    dr_syst_u=atom.charge_radius_FB_uncertainty(ai,Z,qi,N+1,R,cov_ai_syst_u)
    dr_syst_l=atom.charge_radius_FB_uncertainty(ai,Z,qi,N+1,R,cov_ai_syst_l)
    dr_model=atom.charge_radius_FB_uncertainty(ai,Z,qi,N+1,R,cov_ai_model)
    #dr_model_u=atom.charge_radius_FB_uncertainty(ai,Z,qi,N+1,R,cov_ai_model_u)
    #dr_model_l=atom.charge_radius_FB_uncertainty(ai,Z,qi,N+1,R,cov_ai_model_l)
    #
    db_syst=atom.Barrett_moment_FB_uncertainty(ai,Z,qi,R,k,alpha,cov_ai_syst)
    db_syst_u=atom.Barrett_moment_FB_uncertainty(ai,Z,qi,R,k,alpha,cov_ai_syst_u)
    db_syst_l=atom.Barrett_moment_FB_uncertainty(ai,Z,qi,R,k,alpha,cov_ai_syst_l)
    db_model=atom.Barrett_moment_FB_uncertainty(ai,Z,qi,R,k,alpha,cov_ai_model)
    #db_model_u=atom.Barrett_moment_FB_uncertainty(ai,Z,qi,R,k,alpha,cov_ai_model_u)
    #db_model_l=atom.Barrett_moment_FB_uncertainty(ai,Z,qi,R,k,alpha,cov_ai_model_l)
    #
    syst_value_uncert_dict={'dr_ch_syst':dr_syst,'dr_ch_syst_upper':dr_syst_u,'dr_ch_syst_lower':dr_syst_l,'dr_ch_model':dr_model,'dbarrett_syst':db_syst,'dbarrett_syst_upper':db_syst_u,'dbarrett_syst_lower':db_syst_l,'dbarrett_model':db_model}
    #
    best_fit={**best_fit,**syst_value_uncert_dict}
    #
    # naive systematic from distance
    #
    r=best_fit['r_ch']
    b=best_fit['barrett']
    #
    rmax=r
    rmin=r
    #
    bmin=b
    bmax=b
    #
    for key in best_fits:
        ri = best_fits[key]['r_ch']
        if ri>rmax:
            rmax=ri
        elif ri<rmin:
            rmin=ri
        #
        bi = best_fits[key]['barrett']
        if bi>bmax:
            bmax=bi
        elif bi<bmin:
            bmin=bi
    #
    dr_dist_u = rmax-r
    dr_dist_l = r-rmin
    dr_dist = np.max([dr_dist_u,dr_dist_l],axis=0)
    #
    db_dist_u = bmax-b
    db_dist_l = b-bmin
    db_dist = np.max([db_dist_u,db_dist_l],axis=0)
    #
    dist_value_uncert_dict={'dr_ch_dist':dr_dist,'dr_ch_dist_upper':dr_dist_u,'dr_ch_dist_lower':dr_dist_l,'dbarrett_dist':db_dist,'dbarrett_dist_upper':db_dist_u,'dbarrett_dist_lower':db_dist_l}
    #
    best_fit={**best_fit,**dist_value_uncert_dict}
    #
    best_fit={**best_fit,**syst_param_uncert_dict}
    #
    return best_fit