import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import hierarchical_bootstrap.bootstrap as hb
import hierarchical_bootstrap.make_dataset as md

'''
   Development functions for comparing implementations 
'''

def run_test(version,n,nboots,levels=['level_1']):
    '''
        Compute bootstraps for a single version, data structure, and nboots
    '''    

    df = md.make_data(n=n)
    start = time.time()
    bootstraps = hb.bootstrap(df,levels=levels,nboots=nboots,version=version)
    end = time.time()
    duration = end-start

    test = {
        'version':version,
        'nboots':nboots,
        'n':n,
        'duration':duration,
        'data-points':len(df)
    }
    test['time/boot'] = test['duration']/nboots
    test['time/data-point'] = test['duration']/len(df)
    test['time/branching'] = test['duration']/n[0]
    return test
    
def test_nboots(version,n,nboots=[10,100,1000],levels=['level_1']):
    '''
        Scale linearly with number of bootstraps
    '''
    tests = []
    for nboot in nboots:
        tests.append(run_test(version,n,nboot,levels=levels))
    
    return pd.DataFrame(tests)

def test_n(version,nboots=100,levels=['level_1']):
    ns = [[10,10,10],[20,20,20],[30,30,30]]
    tests = []
    for n in ns:
        tests.append(run_test(version,n,nboots,levels=levels))
    return pd.DataFrame(tests)

def check_versions(versions=['1','2'],nboots=100,levels=['level_1']):
    '''
        Quick sanity check that the implementations agree by comparing the quantiles of the
        bootstrapped samples
    '''
    # Make dataset
    df = md.make_data()
    
    # Perform bootstraps
    x1 = hb.bootstrap(df,nboots=nboots,version=versions[0],levels=levels)
    x2 = hb.bootstrap(df,nboots=nboots,version=versions[1],levels=levels)
    x = np.sort(x1['response'])
    y = np.sort(x2['response'])

    # Plot quantile-quantile plot
    fig,ax = plt.subplots()
    quantiles = np.linspace(start=0,stop=1,num=int(np.round(nboots/20)))
    x_quantiles = np.nanquantile(x,quantiles, interpolation='linear')[1:-1]
    y_quantiles = np.nanquantile(y,quantiles, interpolation='linear')[1:-1]
    max_val = np.max([np.max(x_quantiles),np.max(y_quantiles)])
    min_val = np.min([np.min(x_quantiles),np.min(y_quantiles)])
    ax.plot(x_quantiles, y_quantiles, 'o-',alpha=.5)
    ax.plot([min_val,max_val],[min_val,max_val],'k--',alpha=.5)

