import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')



########## Load data ###
train2 = pd.read_csv('train_data_bank.csv')
test2 = pd.read_csv('test_data_bank.csv')

########## 4. Data Exploration and Transformation ###

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
                College = 10th, 11th, 12th, HS-grad
                Degree = Bachelors, Masters, Some-college
                Docs = Doctorate, Prof-school, Assoc-acdm, Assoc-voc
- Marital.Status - 7 categories, probably those without spouses are more willing to be in rich group.
                Divide into new categories: 
                Family = Marries-civ-spouse, Married-AF-spouse
                Single = Never-married, Divorced, Separated, Widowed, Married-spouse-absent
- Occupation - have some nulls, 14 categories,
                Put all categories under 5% into one group
- Relationship - 6 categories, the information os not consistent with Marital.status plus lacking the explanation of 
                the categories meaning I decide to exclude this variable from the model 
                (ideally I would read the spisification for obstaining the data from respondents or
                run two models, one with Marital.status and the other with Relationship)
- Race - 5 categories, top1 (White) is 85% of data
                Put the last 3 in one group 'Other'
- Sex - 2 categories, Women are more likely to be in poor group. Leave as it is.
- Hours.Per.Week - Most people work around 40 hours.
                Make bins: below 40, equals 40 and above 40
- Native.Country - have some nulls, 41 categories, 90% from USA, 2% Mexico, 1.8% null
                    Try making two categories: USA and Others, but it may be that nationality will not be relevant variable
- Income.Group - TARGET
"""


# take care of nulls in Workclass, Occupation ns Native.Country