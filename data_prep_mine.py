import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np



########## Load data ###
train2 = pd.read_csv('train_data_bank.csv')
test2 = pd.read_csv('test_data_bank.csv')

########## 4. Data Exploration ###

train2.columns
train2.head()
train2.info()
train2.describe()
train2.describe(include=['O'])

# convert Income.Group to be boolean
d = {'<=50K': True, '>50K': False}
train2['Income.Group'] = train2['Income.Group'].map(d)

##### AGE #####

train2['Age'].plot.hist()
plt.savefig('age_hist.png')

poor = train2[train2['Income.Group'] == True]
rich = train2[train2['Income.Group'] == False]
poor['Age'].plot.hist(alpha=0.5, color='orange')
rich['Age'].plot.hist(alpha=0.5, color='green')
plt.legend(['Poor', 'Rich'])
plt.savefig('age_hist.png')


##### WORKCLASS #####

# check freq of specific values within variable
train2['Workclass'].value_counts()
train2['Workclass'].value_counts()/train2.shape[0]

# pivot table and plot
workclass_pivot = train2.pivot_table(index='Workclass', values='Income.Group')
workclass_pivot.plot.bar(figsize=(10, 12))
plt.savefig('workclass_pivot.png')

# cross-tab and plot (add percent)
workclass_ct = pd.crosstab(train2['Workclass'], train2['Income.Group'], margins=True)
workclass_ct.iloc[:-1,:-1].plot.bar(figsize=(10, 12), stacked=True, color=['green', 'orange'], grid=False)
plt.savefig('workclass_incomegroup_cross_bar.png')

def percConvert(ser):
    return ser/float(ser[-1])

workclass_ct_perc = workclass_ct.apply(percConvert, axis=1)
workclass_ct_perc.iloc[:-1,:-1].plot.bar(figsize=(10, 12), stacked=True, color=['blue', 'pink'], grid=False)
plt.savefig('workclass_incomegroup_cross_bar_perc.png')




##### EDUCATION #####

train2['Education'].value_counts()
train2['Education'].value_counts()/train2.shape[0]

education_pivot = train2.pivot_table(index='Education', values='Income.Group')
education_pivot.plot.bar(figsize=(10, 12))
plt.savefig('education_pivot.png')


##### MARTIAL STATUS #####

train2['Marital.Status'].value_counts()/train2.shape[0]

mstatus_pivot = train2.pivot_table(index='Marital.Status', values='Income.Group')
mstatus_pivot.plot.bar(figsize=(10, 12))
plt.savefig('maritalstatus_pivot.png')

##### OCCUPATION #####

train2['Occupation'].value_counts()/train2.shape[0]

occ_pivot = train2.pivot_table(index='Occupation', values='Income.Group')
occ_pivot.plot.bar(figsize=(10, 12))
plt.savefig('occupation_pivot.png')


##### RELATIONSHIP #####
train2['Relationship'].value_counts()/train2.shape[0]

rel_pivot = train2.pivot_table(index='Relationship', values='Income.Group')
rel_pivot.plot.bar(figsize=(10, 12))
plt.savefig('relationship_pivot.png')


##### RACE #####
train2['Race'].value_counts()/train2.shape[0]

race_pivot = train2.pivot_table(index='Race', values='Income.Group')
race_pivot.plot.bar(figsize=(10, 12))
plt.savefig('race_pivot.png')


##### SEX #####
train2['Sex'].value_counts()/train2.shape[0]

sex_pivot = train2.pivot_table(index='Sex', values='Income.Group')
sex_pivot.plot.bar(figsize=(10, 12))
plt.savefig('sex_pivot.png')


##### HOURS PER WEEK #####

train2['Hours.Per.Week'].plot.hist()
plt.savefig('hours_hist.png')

poor = train2[train2['Income.Group'] == True]
rich = train2[train2['Income.Group'] == False]
poor['Hours.Per.Week'].plot.hist(alpha=0.5, color='orange')
rich['Hours.Per.Week'].plot.hist(alpha=0.5, color='green')
plt.legend(['Poor', 'Rich'])
plt.savefig('hours_hist.png')


##### NATIVE COUNTRY #####

train2['Native.Country'].value_counts()/train2.shape[0]

sex_pivot = train2.pivot_table(index='Sex', values='Income.Group')
sex_pivot.plot.bar(figsize=(10, 12))
plt.savefig('sex_pivot.png')


##### HANDLING NANS #####
work_nans = list(np.where(train2['Workclass'].isnull())[0])
occ_nans = list(np.where(train2['Occupation'].isnull())[0])

diff = lambda l1,l2: [x for x in l1 if x not in l2]
diff(work_nans, occ_nans)

list(set(work_nans)-set(occ_nans))


"""
Comment: 
- ID is not relevant feature to include in the model
- Age - it seems there are more poor people in the younger age values, 
                But it's worth to put the variable into bins and check again: 0-30, 30-45, 45-60, above 60
- Workclass - have some nulls, there are 8 categories, but 70% of data is one category (Private). 
                Think about combining into new categories, for example: Private, Self-emp, Gov, No-Pay
- Education - 16 categories, where top 3 accounts for 70% of records (HS-grad, Some-college, Bachelors)
                It'd be good to create new categories, which reflects level of education better and are relevant:
                Primary = Preschool, 1st-4th, 5th-6th
                Secondary = 7th-8th, 9th
                College = 10th, 11th, 12th, HS-grad, Some-college
                Degree = Bachelors, Masters, 
                Docs = Doctorate, Prof-school, Assoc-acdm, Assoc-voc
- Marital.Status - 7 categories, probably those without spouses are more willing to be in rich group.
                Divide into new categories: 
                Family = Marries-civ-spouse, Married-AF-spouse
                Single = Never-married, Divorced, Separated, Widowed, Married-spouse-absent
- Occupation - have some nulls, 14 categories,
                Put all categories under 5% into one group
- Relationship - 6 categories, the information os not consistent with Marital.status plus lacking the explanation of 
                the categories meaning I decide to exclude this variable from the model 
                (ideally I would read the specification for obtaining the data from respondents or
                run two models, one with Marital.status and the other with Relationship)
- Race - 5 categories, top1 (White) is 85% of data
                Put the last 3 in one group 'Other'
- Sex - 2 categories, Women are more likely to be in poor group. Leave as it is.
- Hours.Per.Week - Most people work around 40 hours.
                Make bins: below 40, equals 40 and above 40
- Native.Country - have some nulls, 41 categories, 90% from USA, 2% Mexico, 1.8% null
                    Try making two categories: USA and Others, but it may be that nationality will not be relevant variable
- Income.Group - TARGET
- Nans - both Workclass and Occupation variable have around 5% of missing data. 
        It happens that the same people lack information about both. 
        We can either try to replace those Nans or exclude those records completely from analysis.
        I'm going to try option with creating a new category for those missing values ('No value')
"""

##### OUTLIER TREATMENT #####

train2.plot.scatter('ID', 'Age')
plt.savefig('age_scatter.png')

train2.plot.scatter('ID', 'Hours.Per.Week')
plt.savefig('hours_scatter.png')


########## 4. Data Transformation ###

# NaNs in Workclass and Occupation
train_clean = train2.copy()
test_clean = test2.copy()

train_clean['Workclass'] = train_clean['Workclass'].replace(np.nan, 'Missing', regex=True)
train_clean['Occupation'] = train_clean['Occupation'].replace(np.nan, 'Missing', regex=True)

# Age - bins

def put_into_bins(df_train, df_test, variable_label, cut_points, labels):
    """
    Takes data frames with train and test values and put values into bins for a chosen continuous variable
    :param df_train: data frame with train data
    :param df_test: data frame with test data
    :param variable_label: name of variable with values to put into bins
    :param cut_points: list of values to determine bins' borders
    :param labels: list of bin's labels
    :return: two data frames with new columns added
    """
    new_label = variable_label+'_cat'
    df_train[new_label] = pd.cut(df_train[variable_label], cut_points, labels=labels)
    df_test[new_label] = pd.cut(df_test[variable_label], cut_points, labels=labels)

put_into_bins(train_clean, test_clean, 'Age', [0, 30, 45, 60, 100], ['Young', 'Middle Lower', 'Middle Upper', 'Senior'])

age_pivot = train_clean.pivot_table(index='Age_cat', values='Income.Group')
age_pivot.plot.bar(figsize=(10, 12))
plt.savefig('age_pivot.png')


# Workclass - new category (Private, Self-emp, Gov, No-Pay)

def group_with_dict(df_train, df_test, variable_label, cat_dict):
    """
    Takes data frames with train and test values and changes categories' names with those from a dictionary
    for a chosen variable
    :param df_train: data frame with train data
    :param df_test: data frame with test data
    :param variable_label: name of variable with categories to rename
    :param cat_dict: dictionary with new ategories names
    :return: two data frames with grouped categories
    """
    df_train[variable_label] = df_train[variable_label].map(cat_dict)
    df_test[variable_label] = df_test[variable_label].map(cat_dict)

group_with_dict(train_clean, test_clean, 'Workclass', {'Private':'private','Missing':'missing',
                                                       'Self-emp-not-inc':'self_employ', 'Self-emp-inc':'self_employ',
                                                       'Federal-gov':'gov', 'State-gov':'gov',
                                                       'Local-gov':'gov', 'Never-worked':'no_pay',
                                                       'Without-pay':'no_pay'})

# Education - new categories
group_with_dict(train_clean, test_clean, 'Education', {'HS-grad':'college', 'Some-college':'college',
                                                       'Bachelors':'degree', 'Masters':'degree', 'Assoc-voc':'docs',
                                                       '11th':'college', 'Assoc-acdm':'docs', '10th':'college',
                                                       '7th-8th':'secondary', 'Prof-school':'docs', '9th':'secondary',
                                                       '12th':'college', 'Doctorate':'docs', '5th-6th':'primary',
                                                       '1st-4th':'primary', 'Preschool':'primary'})


# Marital.Status - new categories:
group_with_dict(train_clean, test_clean, 'Marital.Status', {'Married-civ-spouse':'family', 'Never-married':'single',
                                                            'Divorced':'single', 'Separated':'single',
                                                            'Widowed':'single', 'Married-spouse-absent':'single',
                                                            'Married-AF-spouse':'family'})


# Occupation - put all categories under 5% into one group

def group_minors(df_train, df_test, variable_label, threshold):
    """
    Takes data frames with train and test values and changes categories' names with frequency below given value
    to 'others' for a chosen variable
    :param df_train: data frame with train data
    :param df_test: data frame with test data
    :param variable_label: name of variable with categories to rename
    :param threshold: frequency value below which all categories will be renamed
    :return: two data frames with combined values in column 'variable_label'
    """
    cat_freq = df_train[variable_label].value_counts()/df_train.shape[0]
    cat_to_combine_index = cat_freq.loc[cat_freq.values < threshold].index
    for category in cat_to_combine_index:
        df_train[variable_label].replace({category:'others'}, inplace=True)
        df_test[variable_label].replace({category:'others'}, inplace=True)

group_minors(train_clean, test_clean, 'Occupation', 0.05)


# Race - Put the last 3 in one group 'Other'
group_with_dict(train_clean, test_clean, 'Race', {'White':'white', 'Black':'black', 'Asian-Pac-Islander':'other',
                                                  'Amer-Indian-Eskimo':'other', 'Other':'other'})


# Hours.Per.Week - Most people work around 40 hours.Make bins: below 40, equals 40 and above 40
put_into_bins(train_clean, test_clean, 'Hours.Per.Week', [0, 39, 40, 100], ['below_mode', 'equals_mode', 'above_mode'])

hours_pivot = train_clean.pivot_table(index='Hours.Per.Week_cat', values='Income.Group')
hours_pivot.plot.bar(figsize=(10, 12))
plt.savefig('hours_pivot.png')


# Native.Country - Try making two categories: USA and Others
country_cat_dict = {}
diff = lambda df: ['other' for x in df if x != 'United-States']
train_clean['Native.Country'] = diff(train_clean['Native.Country'])


for column in cat_variables_for_combining:
    frq = train[column].value_counts()/train.shape[0]
    cat_to_combine = frq.loc[frq.values<0.05].index
    for category in cat_to_combine:
        train[column].replace({category:'Others'}, inplace=True)
        test[column].replace({category:'Others'}, inplace=True)
