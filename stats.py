import numpy as np
import pandas as pd
import scipy.stats as stats

def compute_stats(bootstraps):
    '''
        bootstaps is a dictionary output of bootstrap.bootstrap()
    '''

    # Cannot compare groups if we only have one group
    if len(bootstraps['groups']) == 1:
        print('Only one group, cannot do statistics')
        return

    # Iterate through all pairs of groups and determine p-value
    # by percentage of samples that are greater than the other
    tests = []
    groups = bootstraps['groups']
    for index, group1 in enumerate(groups):
        for jdex,group2 in enumerate(groups[(index+1):]):
            name='{}_{}'.format(group1,group2)
            tests.append({
                'name':name,
                'group1':group1,
                'group2':group2,
                'p':bootstrap_significance(bootstraps,group1,group2),
                'nboots':len(bootstraps[group1])
                })

    return pd.DataFrame(tests)

def bootstrap_significance(bootstrap, k1, k2):
    '''
        Make arbitrary pairings of samples from each group and compute p-val
    '''
    diff = np.array(bootstrap[k1]) - np.array(bootstrap[k2])
    p = np.sum(diff >= 0)/len(diff)
    
    # Flip such that p < 0.5
    if p > 0.5:
        p = 1-p

    return p

def compute_naive_ttest(df,bootstraps,stats_df):
    '''
        Computes a naive t-test that ignores the nested structure.
        Just for demonstration purposes. 

        Run this after compute_stats()
        stats_df = compute_stats(bootstraps)
        stats_df = compute_naive_ttest(df,bootstraps,stats_df)
    '''

    if len(bootstraps['groups']) == 1:
        print('Only one group, cannot do statistics')
        return

    ttests = []
    for index, row in stats_df.iterrows():
        a = df.query('group == @row.group1')['response']
        b = df.query('group == @row.group2')['response']
        ttest = stats.ttest_ind(a,b).pvalue
        ttests.append(ttest)
    stats_df['naive_pvalue'] = ttests


    return stats_df

