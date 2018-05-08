import pandas as pd
from matplotlib import pyplot as plt


### Load data ###
train = pd.read_csv('train_data_bank.csv')
test = pd.read_csv('test_data_bank.csv')

### 4.2. Univariate analysis ###

# 4.2.1 for continuous variables:
train.dtypes
train.describe()
# IQR (interquartile range) is a difference between 1st and 3rd quartile, which is 48-28=20 with the Age variable

# 4.2.2 for categorical variables:
#.1 get list of categorical variables
categorical_variables = train.dtypes.loc[train.dtypes=='object'].index
print(categorical_variables)

#.2 determine the number of unique values in each column
train[categorical_variables].apply(lambda x: len(x.unique()))

#.2.1 Analyze Race (show how many values are in each category and the % of all observations)
train['Race'].value_counts()
train['Race'].value_counts()/train.shape[0]
# Comment: There are 5 categories. The most popular category accounts for 85% and top two combined have around 95% of observations

#.2.2 Analyze Native-Country (show how many values are in each category and the % of all observations)
train['Native.Country'].value_counts()
train['Native.Country'].value_counts()/train.shape[0]
# Comment: There are 42 countries. The most popular accounts for almost 90% of data, the rest os very granular,
#          only one from the other 41 countries makes more than 1%.


### 4.3. Multi-variate analysis ###

# 4.3.1.both variables categorical:
ct = pd.crosstab(train['Sex'], train['Income.Group'], margins=True)
print(ct)

#.1 print and plot cross tab
ct.iloc[:-1,:-1].plot.bar(stacked=True, color=['red', 'blue'], grid=False)
plt.savefig('sex_incomegroup_multi_bar.png')

#.2 try % of total
def percConvert(ser):
    return ser/float(ser[-1])

ct2 = ct.apply(percConvert, axis=1)
ct2.iloc[:-1,:-1].plot.bar(stacked=True, color=['red', 'blue'], grid=False)
plt.savefig('sex_incomegroup_multi_bar_perc.png')


# 4.3.2.both variables continuous:


# 4.3.3.one variable categorical and one continuous: