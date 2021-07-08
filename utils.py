#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import sys

#sys.path.append('..')

def backprop(x, **kwargs):
    for key in kwargs.keys():
        try:
            values = kwargs[key]
            x[values] = x[values] + x[key]/len(values)  
            x[key] = 0
        except KeyError:
            i = len(values)
            while i != 0:
                i -= 1
                x = backprop(x,**{key:values[i]})    
    return x      

def forward(x, **kwargs):
    for key in kwargs.keys():
        values = kwargs[key]
        try:
            minval = np.min(x[values])
            x[values] = x[values] - minval
            x[key] = x[key] + len(values)*minval
        except KeyError:
            i = len(values)
            while i != 0:
                i -= 1
                x = forward(x,**{key:values[i]})       
    return x

def add_pairs(X, **kwargs):
    for key in kwargs.keys():
        values = kwargs[key]
        X[key] = X[values[0]] + X[values[1]]
        X = X.drop(columns = [values[0],values[1]])
    return X
    
def del_pairs(x, **kwargs):
    for p in kwargs.keys():
        if x[p] != 0:
            x[p[1]] =  x[p[1]] + x[p] #direction
            x[p[0]] =  x[p[0]] + x[p] #mass1
            x[p[2]] =  x[p[2]] + x[p] #mass2
            x[p] = 0
    return x 

def round_to_int(x,**kwargs):
    for key in kwargs.keys():
        values = kwargs[key]
        m1 = values[0]
        m2 = values[1]
        koef = [0.25, 0.5]
        for k in koef:
            if (x[m1]-k) == np.floor(x[m1]) or (x[m2]+k) == np.ceil(x[m2]):
                x[m1] = x[m1] - k
                x[m2] = x[m2] + k
            elif (x[m1]+k) == np.ceil(x[m1]) or (x[m2]-k) == np.floor(x[m2]):
                x[m1] = x[m1] + k
                x[m2] = x[m2] - k   
    return x

def decomposition(out_y):
    pairs = {'AXBX':['AX','BX'],'BYCY':['BY','CY'],'CXDX':['CX','DX'],'DYAY':['DY','AY']}
    directQL = {'Q': [['AQ','CQ'],['BL','DL']],'L':[['AL','CL'],['BQ','DQ']]}
    massQL = {'A': ['AQ','AL'],'B': ['BQ','BL'],'C': ['CQ','CL'], 'D': ['DQ','DL']}
    delete = {'del': [['X','Y'],['Q','L']]}
    out_y[['A','B','C','D','X','Y','Q','L','del']] = 0
    
    out_y = add_pairs(out_y, **pairs) #from component mass(X,Y) to pairs (AXBX, CXDX, BYCY, DYAY)     
    out_y = out_y.apply(del_pairs, **pairs, axis = 1) # from pairs (AXBX,CXDX,BYCY,DYAY) to mass(A,B,C,D) and direction (X,Y)
    out_y = out_y.apply(backprop, **massQL, axis = 1) # from mass(A,B,C,D) to component mass(Q,l)
    out_y = out_y.apply(forward, **directQL, axis = 1) # from component mass (Q,L) to direction (Q,L)
    out_y = out_y.apply(forward, **delete, axis = 1) # delete bias
    out_y = out_y.drop(columns = ['AXBX','CXDX','BYCY','DYAY','A','B','C','D','del'])
    return out_y

def integration(out_y, nelem, koefXY = 1, koefQL = 1):
    rounding = 0
    if koefXY != 1:
        rounding = 2  
    pairs = {'AXBX':['AX','BX'],'BYCY':['BY','CY'],'CXDX':['CX','DX'],'DYAY':['DY','AY']}
    directXY = {'X': ['AXBX','CXDX'],'Y': ['BYCY','DYAY']}
    directQL_sum = {'Q': ['AQ','CQ','BL','DL'],'L':['AL','CL','BQ','DQ']}
    massQL = {'A': ['AQ','AL'],'B': ['BQ','BL'],'C': ['CQ','CL'], 'D': ['DQ','DL']}
    massXY = {'A': ['AXBX','DYAY'],'B': ['AXBX','BYCY'],'C': ['CXDX','BYCY'], 'D': ['CXDX','DYAY']}
    out_y[['AX','BX','BY','CY','CX','DX','DY','AY','AXBX','CXDX','BYCY','DYAY','A','B','C','D']] = 0
    
    out_y = out_y.apply(forward, **massQL, axis = 1)
    out_y = out_y.apply(backprop, **directXY, axis = 1)
    out_y = out_y.apply(backprop, **directQL_sum, axis = 1)
    out_y = out_y.round(rounding)
    out_y = out_y.apply(backprop, **massXY, axis = 1)
    out_y = out_y.round(rounding)
    out_y = out_y.apply(backprop, **pairs, axis = 1)
    out_y = out_y[['AX','BX','BY','CY','CX','DX','DY','AY','AQ','AL','BQ','BL','CQ','CL','DQ','DL']]
    out_y = out_y.apply(round_to_int, **pairs, axis = 1)
    out_y = out_y.round(rounding)
    numb = np.where(out_y>nelem)[0]
    numb = np.unique(numb)
    #out_y = out_y.drop(numb)
    out_y = out_y.reset_index(drop=True)
    out_y.iloc[:,:8] = koefXY*out_y.iloc[:,:8]
    out_y.iloc[:,8:] = koefQL*out_y.iloc[:,8:]
    return out_y

def create_out(len_out: int = 100):
    in_y = np.random.uniform(0, 28, [len_out,16])
    #in_y = np.abs(np.random.normal(0,10, size = (len_out,16)))
    in_y = in_y.round(0)
    in_y[in_y > 28] = 28
    columns_y = ['AX', 'BX', 'BY', 'CY', 'CX', 'DX', 'DY', 'AY', 'AQ', 'AL', 'BQ', 'BL', 'CQ', 'CL', 'DQ', 'DL']
    in_y = pd.DataFrame(in_y, columns = columns_y)
    in_y = decomposition(in_y)
    in_y = integration(in_y)
    out_y = in_y.astype('str')
    out_y['AX'] = 'numb_dele(1) = ' + out_y['AX']
    eof = pd.DataFrame(columns = out_y.columns)
    out_y.loc[1::2] = '/eof'
    out_y.to_csv('rand_y.txt',index = False, index_label = False)
    return in_y

def add_phase(x, **kwargs):
    for key in kwargs.keys():
        values = kwargs[key]
        dir1 = values[0]
        dir2 = values[1]
        if x[dir2] == 0 or x[dir1] == 0:
            x[key] = 0.5
        elif (np.sign(x[dir1]) == np.sign(x[dir2])):
            x[key] = np.sin(np.arctan(x[dir2]/x[dir1]) * 1.33 + 0.523)
        elif (np.sign(x[dir1]) != np.sign(x[dir2])):
            x[key] = 1 -  np.sin(np.arctan(np.abs(x[dir2]/x[dir1])) * 1.33 + 0.523)
    return x

def init_in(X, freq_bias):  
    nmode = len(freq_bias) 
    ind1_list = []
    ind2_list = []
    for i in range(X.shape[0]):
        ind1_list.append(i//nmode) 
        ind2_list.append(i%nmode+1)
    mInd = pd.MultiIndex.from_arrays([ind1_list, ind2_list], names=('number', 'mode'))
    X.index = mInd
    
    phase = {'phaseA':['AX','AY'],'phaseB':['BX','BY'],'phaseC':['CX','CY'],'phaseD':['DX','DY']}
    X = X.apply(add_phase, **phase, axis = 1)
    
    pairs = {'AXBX':['AX','BX'],'BYCY':['BY','CY'],'CXDX':['CX','DX'],'DYAY':['DY','AY']}
    X = np.abs(X)
    X = add_pairs(X, **pairs)
    
    pairs_keys = ['AXBX','BYCY','CXDX','DYAY']
    maxval = np.max(X[pairs_keys], axis = 1)
    X[pairs_keys] = X[pairs_keys].divide(maxval, axis = 0, level=1)
    X = X.round(3)
    #for col_name in X.columns:
    #  X[col_name][X[col_name] < 0.2] = 0
    
    #X  = X.groupby(level = 'number').apply(norm_freq, freq_bias)
    

    new_X = pd.DataFrame()
    colX = X.columns
    for idx, mode in enumerate(X.groupby(['mode'])):
        col = {key: key + str(idx+1)  for key in colX}
        new_X = pd.concat([new_X, pd.DataFrame(mode[1]).droplevel(1).rename(columns = col)] , axis = 1) 
    
    for i in range(1, len(freq_bias) + 7):
        new_X['freq'+ str(8 + i)] = new_X['freq'+ str(i + 1)] - new_X['freq'+ str(i)] 
    
    for i in range(1, len(freq_bias) + 7):
        name = 'freq' + str(i + 1)
        if i < 9:
            new_X[name] = new_X[name] - new_X['freq1']
        new_X[name] = (new_X[name] - new_X[name].min())/(new_X[name].max() - new_X[name].min())
    new_X = new_X.drop(columns = ['freq1'])
    new_X = new_X.round(3)
    #new_X = new_X.drop(columns = ['freq1','freq4'])
#     if (nmode == 8):
#         directXY = {'X2': ['AXBX2','CXDX2'],'Y2': ['BYCY2','DYAY2'],'X3': ['AXBX3','CXDX3'],'Y3': ['BYCY3','DYAY3']}
#         mergeX = {'X':['X2','Y3'],'Y':['Y2','X3']}
#         new_X = add_pairs(new_X, **directXY)/2
#         new_X = add_pairs(new_X, **mergeX)/2
    return new_X

def norm_freq(x, freq_bias): 
    x['freq'] = x['freq'] + freq_bias
    f_deltamin = x['freq'] - np.min(x['freq'])
    f_norm = f_deltamin/np.max(f_deltamin)
    x['freq'] = f_norm.round(3)
    return x  

def search_close(X,  threshold = 0.05):
    ind = []
    for i in range(X.index[-1][0]):
        Xstart = X.loc[i].drop(columns = 'freq')
        for j in (range((i + 1),X.index[-1][0])):
            Xnext = X.loc[j].drop(columns = 'freq')
            if (np.mean(np.abs(Xnext - Xstart), axis = 1) < threshold).all():
                ind.append([i,j]) 
    return ind
                 
 