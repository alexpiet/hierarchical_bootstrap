import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

def bootstrap(df,metric='response', top_level=None, levels=['level_1','level_2'],
    nboots=100,version='3'):   

    if version == '1':
        return bootstrap_v1(df,metric,top_level,levels,nboots)
    elif version == '2':
        return bootstrap_v2(df,metric,top_level,levels,nboots)
    elif version == '3':
        return bootstrap_v3(df,metric,top_level,levels,nboots)
    else:
        print('unknown version')   

###############################################################################
def bootstrap_v1(df,metric='response', top_level=None, levels=['level_1','level_2'],nboots=100):
    '''
        Computes a hierarchical bootstrap of <metric> across the hierarchy
        defined in <levels>. 
        levels, strings referring to columns in 'df' from highest (coarsest) level to lowest (finest)
        top_level splits the data into multiple groups and performs a bootstrap for each group

        DEVELOPMENT VERSION. DO NOT USE
    '''

    if top_level is None:
        summary = {}
        summary[metric] = []
        # Perform each bootstrap
        for i in tqdm(range(0,nboots),desc=metric):
            sum_val, count = sample_hierarchically(df, metric, levels)
            summary[metric].append(sum_val/count)              
         
    else:
        summary = {}
        groups = df[top_level].unique()
        # Iterate over top level groups
        for g in groups:
            summary[g]= []
            temp = df.query('{} == @g'.format(top_level))
    
            # Perform each bootstrap
            for i in tqdm(range(0,nboots),desc=g):
                sum_val, count = sample_hierarchically(temp, metric, levels)
                summary[g].append(sum_val/count)            
    groups = list(summary.keys())
    
    # Compute SEM for each group by taking standard deviation of bootstrapped means 
    for key in groups:
        summary[key+'_sem'] = np.std(summary[key])
    summary['groups'] = groups
    return summary

def sample_hierarchically(df, metric, levels):
    '''
        Sample the levels of the hierarchy and return the mean.
        For efficiency, we return the running total and number of samples
        instead of the list of actual values
    '''
    if len(levels) == 1:
        # At the bottom level
        sum_val = df[metric].sample(n=len(df),replace=True).sum()
        count = len(df)
        return sum_val, count  
    else:
        # Sample with replacement an equal number of times to how many
        # data points we have at this level
        items = df[levels[0]].unique()     
        n = len(items)
        sum_val = 0
        count = 0
        for i in range(0,n):
            # Sample, then recurse to lower levels
            choice = np.random.choice(items)
            temp = df.query('{} == @choice'.format(levels[0]))
            temp_sum_val, temp_count = sample_hierarchically(temp, metric, levels[1:])
            sum_val +=temp_sum_val
            count += temp_count
        return sum_val, count

###############################################################################
def bootstrap_v2(df,metric='response', top_level=None, levels=['level_1','level_2'],nboots=100):
    '''
        Computes a hierarchical bootstrap of <metric> across the hierarchy
        defined in <levels>. 
        levels, strings referring to columns in 'df' from highest (coarsest) level to lowest (finest)
        top_level splits the data into multiple groups and performs a bootstrap for each group
    
        DEVELOPMENT VERSION, DO NOT USE.
        Faster than v1 because we figure out how many times we need to sample each lower-level,
            which cuts down the number of queries we need to do
    '''

    if top_level is None:
        summary = {}
        summary[metric] = []
        # Perform each bootstrap
        for i in tqdm(range(0,nboots),desc=metric):
            sum_val, count = sample_hierarchically_v2(df, metric, levels,1)
            summary[metric].append(sum_val/count)                        
         
    else:
        summary = {}
        groups = df[top_level].unique()
        # Iterate over top level groups
        for g in groups:
            summary[g]= []
            temp = df.query('{} == @g'.format(top_level))
    
            # Perform each bootstrap
            for i in tqdm(range(0,nboots),desc=g):
                sum_val, count = sample_hierarchically_v2(temp, metric, levels)
                summary[g].append(sum_val/count)            
    groups = list(summary.keys())
    
    # Compute SEM for each group by taking standard deviation of bootstrapped means 
    for key in groups:
        summary[key+'_sem'] = np.std(summary[key])
    summary['groups'] = groups
    return summary

def sample_hierarchically_v2(df,metric,levels,num_samples=1):
    if len(levels) == 1:
        # At the bottom level
        sum_val = df[metric].sample(n=len(df)*num_samples,replace=True).sum()
        count = len(df)*num_samples
        return sum_val, count  
    else:
        # Sample with replacement an equal number of times to how many
        # data points we have at this level
        items = df[levels[0]].unique()     
        n = len(items)
        sum_val = 0
        count = 0
        samples = Counter(np.random.choice(items,size=n*num_samples))
        for sample in samples.keys():
            temp = df.query('{} == @sample'.format(levels[0]))
            temp_sum_val, temp_count = \
                sample_hierarchically_v2(temp, metric, levels[1:],samples[sample])
            sum_val +=temp_sum_val
            count += temp_count
        return sum_val, count
  
###############################################################################
def bootstrap_v3(df,metric='response', top_level=None, levels=['level_1','level_2'],nboots=100):
    '''
        Computes a hierarchical bootstrap of <metric> across the hierarchy
        defined in <levels>. 
        levels, strings referring to columns in 'df' from highest (coarsest) level to lowest (finest)
        top_level splits the data into multiple groups and performs a bootstrap for each group
    
        Faster than version 2 because it uses df[ ] syntax instead of df.query()
    '''

    if top_level is None:
        summary = {}
        summary[metric] = []
        # Perform each bootstrap
        for i in tqdm(range(0,nboots),desc=metric):
            sum_val, count = sample_hierarchically_v3(df, metric, levels,1)
            summary[metric].append(sum_val/count)                        
         
    else:
        summary = {}
        groups = df[top_level].unique()
        # Iterate over top level groups
        for g in groups:
            summary[g]= []
            temp = df.query('{} == @g'.format(top_level))
    
            # Perform each bootstrap
            for i in tqdm(range(0,nboots),desc=g):
                sum_val, count = sample_hierarchically_v3(temp, metric, levels)
                summary[g].append(sum_val/count)            
    groups = list(summary.keys())
    
    # Compute SEM for each group by taking standard deviation of bootstrapped means 
    for key in groups:
        summary[key+'_sem'] = np.std(summary[key])
    summary['groups'] = groups
    return summary

def sample_hierarchically_v3(df,metric,levels,num_samples=1):
    if len(levels) == 1:
        # At the bottom level
        sum_val = df[metric].sample(n=len(df)*num_samples,replace=True).sum()
        count = len(df)*num_samples
        return sum_val, count  
    else:
        # Sample with replacement an equal number of times to how many
        # data points we have at this level
        items = df[levels[0]].unique()     
        n = len(items)
        sum_val = 0
        count = 0
        samples = Counter(np.random.choice(items,size=n*num_samples))
        for sample in samples.keys():
            temp = df[df[levels[0]].values == sample]
            temp_sum_val, temp_count = \
                sample_hierarchically_v3(temp, metric, levels[1:],samples[sample])
            sum_val +=temp_sum_val
            count += temp_count
        return sum_val, count
 
