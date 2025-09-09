import numpy as np
pi = np.pi

def generate_data_tables(database,grid_database=None):
    
    if grid_database is None:
        grid_database=database
    
    Rs=np.array([])
    Ns=np.array([])
    for key in grid_database:
        Ri=grid_database[key]['R']
        Rs=np.append(Rs,Ri)
        Ni=grid_database[key]['N']
        Ns=np.append(Ns,Ni)
    Rs=np.unique(Rs)
    Ns=np.unique(Ns)
    
    statistics_quantities = ['redchisq','chisq','p_val','dof']
    other_quantities = ['R','N','r_ch','barrett']
    
    tables={}
    hist={}
    
    for quantity in statistics_quantities: 
        tables[quantity] = np.ones((len(Rs),len(Ns)))*np.nan 
        hist[quantity] = [] 
    
    tables['AIC'] = np.ones((len(Rs),len(Ns)))*np.nan 
    hist['AIC'] = [] 
    
    for quantity in other_quantities: 
        tables[quantity] = np.ones((len(Rs),len(Ns)))*np.nan 
        hist[quantity] = [] 
    
    for key in database:
        quantities = database[key].keys()
        Ri=database[key]['R']
        Ni=database[key]['N']
        i_R=np.argwhere(Rs==Ri)[0][0]
        i_N=np.argwhere(Ns==Ni)[0][0]
        for quantity in statistics_quantities:
            if quantity in quantities:
                tables[quantity][i_R,i_N]=database[key]['statistics_dict'][quantity]['total']
                hist[quantity].append(database[key][quantity])
        for quantity in other_quantities:
            if quantity in quantities:
                tables[quantity][i_R,i_N]=database[key][quantity]
                hist[quantity].append(database[key][quantity])
        if 'chisq' in quantities:
            tables['AIC'][i_R,i_N] = np.exp(-(1/2)*(tables['chisq'][i_R,i_N]+2*Ni))
            hist['AIC'].append(tables['AIC'][i_R,i_N])
    
    return tables, Rs, Ns, hist