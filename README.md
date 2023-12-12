
# hierarchical_bootstrap
Implementation of a hierarchical bootstrap for pandas dataframes. This package can be used to perform two useful things:
 - Estimate the confidence interval that takes into account the hierarchical nature of the data
 - Perform a statistical test to compare the mean value between two groups that have hierachical structure. 

Everything is generally applicable to any nested dataset, although my use case is neuroscience specific. I made some simple attempts to speed up the implementation, but it is still time consuming. I followed the procedure outlined in: 

> Application of the hierarchical bootstrap to multi-level data in neuroscience (2020). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7906290/

Contact: alexpiet [at] gmail [dot] com 

## Demonstration

I developed a simple demonstration of why a hierarchical approach is important when data is nested. The demonstration function will generate synthetic data with a nested structure. Importantly, there is no difference between the generative means of the two processes, but the variability at the two levels of the nested data means that a naive t-test will return a signficant result, while a hierarchical approach will show no effect. 

In this synthetic dataset there are two groups, colored black and mageneta. Each group has a hierarchical structure. As a specific example we might consider the groups two be an experimental manipulation, and the levels as nested observations (Level 1 could be behavioral sessions, level 2 could be cells, and level 3 is each measurement from a single cell). 

> import hierarchical_bootstrap.make_data as md   
> df,bootstraps, stats_df = md.demonstrate_levels() 


![bootstraps_example](https://user-images.githubusercontent.com/7605170/235807446-a2c5d63d-22be-4573-8af2-090187af4527.png)
![bootstrap_levels](https://user-images.githubusercontent.com/7605170/236035325-40eac912-c4f8-40f0-9e74-f7efe992200c.png)

A naive t-test results in a false positive, whereas the hierarchical sampling approach correctly finds no significant difference between the groups. 

## General use

Organize your data in the pandas "tidy" format, where each row is a single observation. Each level of the nested structure should be defined in a column, as well as any top level groups. The observation variable should be its own column. 

### Estimating the standard error of the mean with nested data
For example if we have a dataframe of observations "response" and nested hierarchies "level_1" and "level_2", then we can compute the bootstraps with different levels of hierachical bootstrapping. You can use the `df` variable from the demonstration script above to follow along these examples. To compute non-hierarchical bootstraps, which is just sampling with replacement from all observations, regardless of hierarchy (sample once):

> import hierarchical_bootstrap.bootstrap as hb   
> bootstraps = hb.bootstrap(df, metric='response',levels=[], nboots=10000)

`bootstraps` is a dictionary with:
 - `<metric>` the sampled value of the observation metric for each bootstrap iteration, will have length of `nboots`
 - `<metric>_sem` the estimated standard error of the mean of the observation metric, computed as the standard deviation of the bootstrapped samples. In this case this should be very close to `df['response'].sem()`
 - `groups`, the list of top-levels groups, in this case just `['response']`

To sample with one level of hierarchy, which means we sample with replacement from elements of "level_1", then sample with replacement from all observations within that element of level_1 (sample twice). Note now that the estimated SEM differs significantly from the naive approach above. 

> import hierarchical_bootstrap.bootstrap as hb   
> bootstraps = hb.bootstrap(df, metric='response',levels=['level_1'], nboots=10000)

To sample with two levels of the hierarchy, which means we sample with replacement from elements of "level_1", then sample with replacement from "level_2" elements within that element of level_1, then finally sample from all observations within that level_1, level_2 element (sample three times).

> import hierarchical_bootstrap.bootstrap as hb   
> bootstraps = hb.bootstrap(df, metric='response',levels=['level_1','level_2'], nboots=10000)

How many nesting steps you should take depends on the variance at each level of your dataset. The code should work for as many levels as you want, but performance will suffer greatly from each additional level. 

### Comparing groups and hypothesis testing

If you have multiple top-level manipulations you want to compare, then you can specify that too:

> bootstraps = hb.bootstrap(df, metric='response',levels=['level_1','level_2'], top_level='group',nboots=10000) 

Now `bootstraps` is a dictionary with two entries for each unique group:
 - `<group label>` the sampled value of the observation metric for each bootstrap iteration, only for this group. This will have level `nboots`.
 - `<group label>_sem` the estimatd standard error of the mean of the observation metric for this group. 

You can perform statistical testing with:

> import hierarchical_bootstrap.stats as stats   
> stats_df = stats.compute_stats(bootstraps)

`stats_df` is a dataframe with one row for each hypothesis test performed. Each row has the following columns:
 - name, which is the combination of the two groups compared
 - p, the fraction of bootstrap iterations where group1 > group2
 - nboots, the number of bootstrap iterations used in the test (inherited from `bootstraps`) 



