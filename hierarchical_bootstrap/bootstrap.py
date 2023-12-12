import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

def bootstrap(df,metric='response', top_level=None, levels=['level_1'],
    nboots=100,version='4'):   
    '''
        Wrapper function that selects which implementation to use. 

        df, a pandas dataframe in the tidy format, with columns <metric>, <levels>, 
            and <top_level> if not None
        metric, the name of the column with the observation variable of interest
        top_level, (optional) the name of the column that defines a top-level 
            experimental manipulation
        levels, a list of column names that define the hierarchical structure of the dataset
            the first entry is the coarsest level
        nboots, int, how many bootstrap iterations to perform
        version, str, which implementation version to use. '4' is the fastest at the moment. 
            Unless you have a good reason, use '4'.
    '''

    ## Input checking
    # Check all columns are present
    if metric not in df.columns:
        raise Exception('metric "{}" not found in dataframe'.format(metric))
    if (top_level is not None) and (top_level not in df.columns):
        raise Exception('top_level grouping "{}" not found in dataframe'.format(top_level))
    for level in levels:
        if level not in df.columns:
            raise Exception('hierarchical level "{}" not found in dataframe'.format(level))

    # Check all columns make sense
    if metric in levels:
        raise Exception('metric "{}" cannot be part of hierarchical levels'.format(metric))
    if (top_level is not None) and (top_level in levels):
        raise Exception('top_level grouping '+\
            '"{} cannot be part of hierarchical levels'.format(metric))
    if metric == top_level:
        raise Exception('metric "{}" cannot be the same as the top_level grouping'.format(metric))

    # Make sure data types make sense
    if not pd.api.types.is_numeric_dtype(df[metric]):
        raise Exception('metric "{}" must be a numerical data type'.format(metric))
    if 'float64' in list(df[levels].dtypes.values.astype(str)):
        raise Exception('hierarchical level "{}" must not be continuously valued'.format(level))

    
    ## Call the appropriate version
    if version == '1':
        return bootstrap_v1(df,metric,top_level,levels,nboots)
    elif version == '2':
        return bootstrap_v2(df,metric,top_level,levels,nboots)
    elif version == '3':
        return bootstrap_v3(df,metric,top_level,levels,nboots)
    elif version == '4':
        return bootstrap_v4(df,metric,top_level,levels,nboots)
    else:
        print('unknown version')   

###############################################################################
def bootstrap_v1(df,metric='response', top_level=None, 
    levels=['level_1','level_2'],nboots=100):
    '''
        Computes a hierarchical bootstrap of <metric> across the hierarchy
        defined in <levels>. 
        levels, strings referring to columns in 'df' from highest (coarsest) level to lowest (finest)
        top_level splits the data into multiple groups and performs a bootstrap for each group

        DEVELOPMENT VERSION. DO NOT USE unless you have a good reason
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

        DEVELOPMENT VERSION. DO NOT USE unless you have a good reason
    '''
    if len(levels) == 0:
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
def bootstrap_v2(df,metric='response', top_level=None, 
    levels=['level_1','level_2'],nboots=100):
    '''
        Computes a hierarchical bootstrap of <metric> across the hierarchy
        defined in <levels>. 
        levels, strings referring to columns in 'df' from highest (coarsest) level to lowest (finest)
        top_level splits the data into multiple groups and performs a bootstrap for each group
    
        DEVELOPMENT VERSION. DO NOT USE unless you have a good reason
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
    '''
    DEVELOPMENT VERSION. DO NOT USE unless you have a good reason
    '''
    if len(levels) == 0:
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
def bootstrap_v3(df,metric='response', top_level=None, 
    levels=['level_1','level_2'],nboots=100):
    '''
        Computes a hierarchical bootstrap of <metric> across the hierarchy
        defined in <levels>. 
        levels, strings referring to columns in 'df' from highest (coarsest) level to lowest (finest)
        top_level splits the data into multiple groups and performs a bootstrap for each group
    
        Faster than version 2 because it uses df[ ] syntax instead of df.query()
        
        DEVELOPMENT VERSION. DO NOT USE unless you have a good reason
    '''

    if top_level is None:
        # We don't have a top level, so make a group named <metric>
        summary = {}
        summary[metric] = []
        # Perform each bootstrap
        for i in tqdm(range(0,nboots),desc=metric):
            sum_val, count = sample_hierarchically_v3(df, metric, levels,1)
            summary[metric].append(sum_val/count)                        
         
    else:
        # We have a top level, so determine how many groups there are
        summary = {}
        groups = df[top_level].unique()

        # Iterate over top level groups
        for g in groups:
            summary[g]= []
            # Filter the data frame to just this group
            temp = df.query('{} == @g'.format(top_level))
    
            # Perform each bootstrap
            for i in tqdm(range(0,nboots),desc=g):
                sum_val, count = sample_hierarchically_v3(temp, metric, levels,1)
                summary[g].append(sum_val/count)      

    # Determine how many groups we had      
    groups = list(summary.keys())
    
    # Compute SEM for each group by taking standard deviation of bootstrapped means 
    for key in groups:
        summary[key+'_sem'] = np.std(summary[key])
    summary['groups'] = groups

    # Return summary dictionary
    return summary

def sample_hierarchically_v3(df,metric,levels,num_samples=1):
    '''
        num_samples, int, how many times to sample this level of the hierarchy
        DEVELOPMENT VERSION. DO NOT USE unless you have a good reason
    '''
    if len(levels) == 0:
        # At the bottom level, just sample the rows
        sum_val = df[metric].sample(n=len(df)*num_samples,replace=True).sum()
        count = len(df)*num_samples
        return sum_val, count  
    else:
        # Sample with replacement an equal number of times to how many
        # data points we have at this level, multiplied by how many times we are sampling
        # this level
        items = df[levels[0]].unique()     
        samples = Counter(np.random.choice(items,size=len(items)*num_samples))

        # Iterate through each sample and recursively compute the bootstrap
        sum_val = 0
        count = 0
        for sample in samples.keys():
    
            # Filter the dataset for this sample
            temp = df[df[levels[0]].values == sample]
        
            # Recursively compute bootstrap
            temp_sum_val, temp_count = \
                sample_hierarchically_v3(temp, metric, levels[1:],samples[sample])

            # Keep track of running sum and running number of datapoints
            sum_val +=temp_sum_val
            count += temp_count
            
        return sum_val, count


###############################################################################
def bootstrap_v4(df,metric='response', top_level=None, 
    levels=['level_1','level_2'],nboots=100):
    '''
        Computes a hierarchical bootstrap of <metric> across the hierarchy
        defined in <levels>. 
        levels, strings referring to columns in 'df' from highest (coarsest) level to lowest (finest)
        top_level splits the data into multiple groups and performs a bootstrap for each group
    
        Faster than version 3 because it computes all the bootstraps in parallel which means
            you only need to perform each query a single time
            (About 4x faster than v3, and 20x faster than v1)
    '''

    if top_level is None:
        # We don't have a top level, so make a group named <metric>
        summary = {}
        summary[metric] = []

        # Perform each bootstrap
        sums, count = sample_hierarchically_v4(df,metric,levels,[1]*nboots,nboots)
        summary[metric] = sums/count
         
    else:
        # We have a top level, so determine how many groups there are
        summary = {}
        groups = df[top_level].unique()

        # Iterate over top level groups
        for g in groups:
            summary[g]= []
            # Filter the data frame to just this group
            temp = df.query('{} == @g'.format(top_level))

            sums, count = sample_hierarchically_v4(temp,metric,levels,[1]*nboots,nboots)
            summary[g] = sums/count   

    # Determine how many groups we had      
    groups = list(summary.keys())
    
    # Compute SEM for each group by taking standard deviation of bootstrapped means 
    for key in groups:
        summary[key+'_sem'] = np.std(summary[key])
    summary['groups'] = groups

    # Return summary dictionary
    return summary

def sample_hierarchically_v4(df,metric,levels,num_samples=[1],nboots=1):
    '''
        num_samples, int, how many times to sample this level of the hierarchy
    '''
    if len(num_samples) != nboots:
        raise Exception('bad input size')
    
    if len(levels) == 0:
        # At the bottom level, just sample the rows
        sums = np.zeros(nboots)
        counts = np.zeros(nboots)
        for i in range(0,nboots):
            if num_samples[i] > 0:
                sums[i] = df[metric].sample(n=len(df)*int(num_samples[i]),replace=True).sum()
                counts[i] = len(df)*num_samples[i]
        return sums, counts  
    else:
        # Sample with replacement an equal number of times to how many
        # data points we have at this level, multiplied by how many times we are sampling
        # this level. We do this for each bootstrap
        items = df[levels[0]].unique()     
        samples = np.zeros((len(items),nboots))
        for i in range(0,nboots): 
            count = Counter(np.random.choice(items,size=len(items)*int(num_samples[i])))
            for index,item in enumerate(items):
                samples[index,i]=count[item]

        # Iterate through each sample and recursively compute the bootstraps
        sums = np.zeros(nboots)
        counts = np.zeros(nboots)
        for index,item in enumerate(items):
    
            # Filter the dataset for this sample
            temp = df[df[levels[0]].values == item]
        
            # Recursively compute an array of bootstrap values for this sample
            temp_sums, temp_counts = \
                sample_hierarchically_v4(temp, metric, levels[1:],samples[index,:],nboots)

            # Keep track of running sum and running number of datapoints
            sums = sums + temp_sums
            counts = counts + temp_counts
            
        return sums, counts



