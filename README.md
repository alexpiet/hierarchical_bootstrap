
# hierarchical_bootstrap
Hierarchical bootstrap for pandas dataframes. Everything is generally applicable to any nested dataset, although my use case is neuroscience specific. I made some simple attempts to speed up the implementation, but it is still time consuming. I followed the procedure outlined in: 

> Application of the hierarchical bootstrap to multi-level data in neuroscience (2020). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7906290/

Contact: alexpiet [at] gmail [dot] com 

## Demonstration

I developed a simple demonstration of why a hierarchical approach is important when data is nested. The demonstration function will generate synthetic data with a nested structure. Importantly, there is no difference between the generative means of the two processes, but the variability at the two levels of the nested data means that a naive t-test will return a signficant result, while a hierarchical approach will show no effect. 

In this synthetic dataset there are two groups, colored black and mageneta. Each group has a hierarchical structure. As a specific example we might consider the groups two be an experimental manipulation, and the levels as nested observations (Level 1 could be behavioral sessions, level 2 could be cells, and level 3 is each measurement from a single cell). 

> import hierarchical_bootstrap.make_data as md   
> df,bootstraps, stats_df = md.demonstration() 


![bootstraps_example](https://user-images.githubusercontent.com/7605170/235807446-a2c5d63d-22be-4573-8af2-090187af4527.png)
![bootstrap_levels](https://user-images.githubusercontent.com/7605170/236035325-40eac912-c4f8-40f0-9e74-f7efe992200c.png)


## General use

Organize your data in the pandas "tidy" format, where each row is a single observation. Each level of the nested structure should be defined in a column, as well as any top level groups. The observation variable should be its own column. For example if we have a dataframe of observations "response" and nested hierarchies "level_1" and "level_2", then we can compute the bootstraps with different levels of hierachical bootstrapping. To compute non-hierarchical bootstraps, which is just sampling with replacement from all observations, regardless of hierarchy (sample once):

> import hierarchical_bootstrap.bootstrap as hb   
> bootstraps = hb.bootstrap(df, metric='response',levels=[], nboots=10000)

To sample with one level of hierarchy, which means we sample with replacement from elements of "level_1", then sample with replacement from all observations within that element of level_1 (sample twice).

> import hierarchical_bootstrap.bootstrap as hb   
> bootstraps = hb.bootstrap(df, metric='response',levels=['level_1'], nboots=10000)

To sample with two levels of the hierarchy, which means we sample with replacement from elements of "level_1", then sample with replacement from "level_2" elements within that element of level_1, then finally sample from all observations within that level_1, level_2 element (sample three times).

> import hierarchical_bootstrap.bootstrap as hb   
> bootstraps = hb.bootstrap(df, metric='response',levels=['level_1','level_2'], nboots=10000)

How many nesting steps you should take depends on the variance at each level of your dataset. The code should work for as many levels as you want, but performance will suffer greatly from each additional level. 

The output variable is a dictionary with three keys. The first <metric> is a list of the bootstrap samples, so a list of length <nboots>. The second <metric>_sem is the estimated hierarchical standard error of the mean, which is computed simply as the standard deviation of the bootstrap samples. The last "groups" is a list of top-level groups, which for this simple case if just length 1. 

If you have multiple top-level manipulations you want to compare, then you can specify that too:

> bootstraps = hb.bootstrap(df, metric='response',levels=['level_1','level_2'], top_level='group',nboots=10000) 

You can perform statistical testing with:

> import hierarchical_bootstrap.stats as stats   
> bootstraps = stats.compute_stats(bootstraps)


