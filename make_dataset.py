import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import hierarchical_bootstrap.bootstrap as hb
import hierarchical_bootstrap.stats as stats

def demonstration_1():

    # Make a synthetic dataset with one overall group (no "top level"), and hierarchical
    # levels "level_1" and "level_2", and the metric of interest "response"
    df1 = make_data()

    # Compute the bootstraps
    bootstraps = hb.bootstrap(df1,levels=['level_1','level_2'],nboots=100)
    
    # Plot the bootstraps
    plot_data(df1,bootstraps)
  
def demonstration(group_diff = .1,nboots=1000,seed=1):
    np.random.seed(seed)
    # Make a synthetic dataset with two overall groups 
    df = make_two_group_data(group_diff=group_diff)
    bootstraps = hb.bootstrap(df,levels=['level_1','level_2'],top_level='group',nboots=nboots)
    stats_df = stats.compute_stats(bootstraps)
    stats_df = stats.compute_naive_ttest(df,bootstraps,stats_df)
    plot_data(df,bootstraps)
    plot_stats_demonstration(df,bootstraps,stats_df)
    print(stats_df)
    return df, bootstraps, stats_df
 

def make_two_group_data(n=[10,10,10],group_diff = 1):
    df1 = make_data(n=n)
    df1['group'] = 'group_1'
    df2 = make_data(n=n)
    df2['group'] = 'group_2'
    df2['response'] = df2['response'] + group_diff
    df = pd.concat([df1,df2])
    return df
    
def make_data(n=[10,10,10]):
    '''
        Generates synthetic data with groups, subjects, cells, and image level sampling
    '''

    level_1_var = 1
    level_2_var = 1
    level_3_var = 1
    dfs = []

    # Iterate over experiments
    for i in range(0,n[0]):
        i_mean = np.random.randn()*level_1_var

        # Iterate over cells
        for j in range(0, n[1]):
            j_mean = i_mean + np.random.randn()*level_2_var
    
            # Iterate over image responses
            for k in range(0,n[2]):
                response = j_mean+np.random.randn()*level_3_var
                this_df = {
                    'level_1':i,
                    'level_2':j,
                    'level_3':k,
                    'response':response
                    }
                dfs.append(pd.DataFrame(this_df,index=[0]))

    # Combine responses
    df = pd.concat(dfs).reset_index(drop=True)
    return df

def plot_data(df,bootstraps):
    plt.figure()
    
    num_groups = len(bootstraps['groups'])
   
    if num_groups == 1: 
        # Grand mean
        m = df['response'].mean()
        sem = df['response'].sem()
        plt.plot(0,m,'ko')
        plt.plot([0,0],[m-sem,m+sem],'k-')
        boot_sem = np.std(bootstraps['response'])
        plt.plot([0,0],[m-boot_sem,m+boot_sem],'m-')
     
        # Plot level 1
        means = df.groupby('level_1')['response'].mean()
        plt.plot([1]*len(means),means,'o')
    
        # Plot level 2
        df['level_1_2'] = df['level_1'].astype(str) + df['level_2'].astype(str)
        means = df.groupby('level_1_2')['response'].mean()
        plt.plot([2]*len(means),means,'o')
    
        # Plot level 3
        plt.plot([3]*len(df),df['response'],'o')
    
        plt.ylabel('value',fontsize=16)
        plt.xticks([0,1,2,3],labels=['mean','level 1','level 2','level 3'],fontsize=12)
    else:
        colors = ['k','m','r','b']
        dx = .1
        for index,group in enumerate(bootstraps['groups']):
            # Grand mean
            group_df = df.query('group ==@group').copy()
            m = group_df['response'].mean()
            sem = group_df['response'].sem()
            plt.plot(index*dx,m,'o',color=colors[index])
            plt.plot([index*dx,index*dx],[m-sem,m+sem],'-',color=colors[index])
            boot_sem = bootstraps[group+'_sem']
            plt.plot([index*dx,index*dx],[m-boot_sem,m+boot_sem],'-',color=colors[index])
         
            # Plot level 1
            means = group_df.groupby('level_1')['response'].mean()
            plt.plot([1+index*dx]*len(means),means,'o',color=colors[index])
        
            # Plot level 2
            group_df['level_1_2'] = group_df['level_1'].astype(str) + group_df['level_2'].astype(str)
            means = group_df.groupby('level_1_2')['response'].mean()
            plt.plot([2+index*dx]*len(means),means,'o',color=colors[index])
        
            # Plot level 3
            plt.plot([3+index*dx]*len(group_df),group_df['response'],'o',color=colors[index])
    
        plt.ylabel('value',fontsize=16)
        plt.xticks([0,1,2,3],labels=['groups','level 1','level 2','level 3'],fontsize=12)           

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def plot_stats_demonstration(df,bootstraps, stats_df):
    colors = ['k','m']
    dx = .1
    plt.figure(figsize=(3,3))
    for index,group in enumerate(bootstraps['groups']):
        # Grand mean
        group_df = df.query('group ==@group').copy()
        m = group_df['response'].mean()

        # Naive approach
        sem = group_df['response'].sem()
        plt.plot(index*dx,m,'o',color=colors[index])
        plt.plot([index*dx,index*dx],[m-sem,m+sem],'-',color=colors[index])

        # hierarchical approach
        boot_sem = bootstraps[group+'_sem']
        plt.plot(1+index*dx,m,'o',color=colors[index])
        plt.plot([1+index*dx,1+index*dx],[m-boot_sem,m+boot_sem],'-',color=colors[index])

    if stats_df.loc[0]['p'] < 0.05:
        plt.plot(1+dx/2,.9,'k*')   
    else:
        plt.text(1,.9, 'ns')
    if stats_df.loc[0]['naive_pvalue'] < 0.05:
        plt.plot(dx/2,.9,'k*') 
    else:
        plt.text(0,.9, 'ns')
 
    plt.xticks([0,1],labels=['naive','hierarchical'],fontsize=12)
    plt.ylabel('value',fontsize=16) 
    plt.xlim(-.5,1.5)
    plt.ylim(-1,1)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()

