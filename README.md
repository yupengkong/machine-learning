# machine-learning project
1. Census Income prediction

This dataset comes from UCI machine learning repository https://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD)
It includes 199,523 observations in the training dataset and 99,762 observations in the test dataset. The task is to
predict whether a person's income is greater than $50K per year with the given 40 features.

There are two challenges of this dataset. One is imbalance (more than 95% of observations is positive). The other is the 
missing values in the features. The basic approach is to use R to train a simple model with decision tree and random forest,
since both of them can handle missing value nicely. In addition, they can provide the metric to measure variable importance.
luckily, all the features containing missing values are not important.Therefore, the missing value problem is solved 
by drop all the features with missing values.

For the issue of imbalance, upsampling was performed to make even number of postive and negative observations. To test
the model, F-1 score was calculated on both major and minor class.

A randomforest model can give a F-1 score of 0.6 for the minor class on the separate test set.
