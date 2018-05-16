import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


########## Load data ###
train = pd.read_csv('train_data_bank.csv')
test = pd.read_csv('test_data_bank.csv')


########## 4.2. Univariate analysis ###

# 4.2.1 for continuous variables:
train.dtypes
train.describe()
# IQR (interquartile range) is a difference between 1st and 3rd quartile, which is 48-28=20 with the Age variable

# 4.2.2 for categorical variables:
#.1 get list of categorical variables
categorical_variables = train.dtypes.loc[train.dtypes == 'object'].index
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


########## 4.3. Multi-variate analysis ###

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
train.plot.scatter('Age', 'Hours.Per.Week')
plt.savefig('age_hours_multi_scatter.png')
# Comment: No obvious trend

# 4.3.3.one variable categorical and one continuous:
train.boxplot(column='Hours.Per.Week', by='Sex')
plt.savefig('sex_hours_multi_box.png')
# Comment: The median is the same, but but men have higher hours in general


########## 4.4. Missing value treatment ###

train.apply(lambda x: sum(x.isnull()))
train.isnull().sum()
test.isnull().sum()
# Comment: we have missing values in 3 variables (all categorical). We're going to try imputation method with mode

var_to_impute = ['Workclass', 'Occupation', 'Native.Country']
for var in var_to_impute:
    train[var].fillna(train[var].mode()[0], inplace=True)
    test[var].fillna(test[var].mode()[0], inplace=True)

train.isnull().sum()
test.isnull().sum()


########## 4.5. Outlier treatment ###
train.plot.scatter('ID', 'Age')
plt.savefig('id_age_scatter.png')

train.plot.scatter('ID', 'Hours.Per.Week')
plt.savefig('id_hours_scatter.png')


########## 4.6. Variable transformation ###

# 4.5.1. workclass - Combine values with less than 5% into one group
train['Workclass'].value_counts()/train.shape[0]

categories_to_combine = ['State-gov', 'Self-emp-inc', 'Federal-gov', 'Without-pay', 'Never-worked']
for category in categories_to_combine:
    train['Workclass'].replace({category:'Others'}, inplace=True)
    test['Workclass'].replace({category:'Others'}, inplace=True)

train['Workclass'].value_counts()/train.shape[0]
test['Workclass'].value_counts()/test.shape[0]

# 4.5.2 Do the same combining for the rest of categorical variables

cat_variables_for_combining = list(categorical_variables)
cat_variables_for_combining.remove('Workclass')

train[cat_variables_for_combining].apply(lambda x: len(x.unique()))

for column in cat_variables_for_combining:
    frq = train[column].value_counts()/train.shape[0]
    cat_to_combine = frq.loc[frq.values<0.05].index
    for category in cat_to_combine:
        train[column].replace({category:'Others'}, inplace=True)
        test[column].replace({category:'Others'}, inplace=True)

train[cat_variables_for_combining].apply(lambda x: len(x.unique()))
test[cat_variables_for_combining[:-1]].apply(lambda x: len(x.unique()))


########## 5. Predictive Modeling - Decision Tree ###

# convert all categorical variables to numerical
le = LabelEncoder()
for variable in categorical_variables:
    train[variable] = le.fit_transform(train[variable])

for variable in categorical_variables[:-1]:
    test[variable] = le.fit_transform(test[variable])

train.dtypes
test.dtypes

# set the predictors
dependent_variable = 'Income.Group'
independent_variables = [x for x in train.columns if x not in ['ID', dependent_variable]]

# initialize and fit the algorithm
model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=100, max_features='sqrt')
model.fit(train[independent_variables], train[dependent_variable])

# predict
predict_train = model.predict(train[independent_variables])
predict_test = model.predict(test[independent_variables])

# evaluate accuracy
acc_train = accuracy_score(train[dependent_variable], predict_train)
accuracy = cross_val_score(model, train[independent_variables], train[dependent_variable], cv=10).mean()
