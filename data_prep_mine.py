import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


########## Load data ###
train2 = pd.read_csv('train_data_bank.csv')
test2 = pd.read_csv('test_data_bank.csv')

# convert Income.Group to be boolean
bool_income_d = {'<=50K': True, '>50K': False}
train2['Income.Group'] = train2['Income.Group'].map(bool_income_d)

########## 4. Data Exploration ###

train2.columns
train2.head()
train2.info()
train2.describe()
train2.describe(include=['O'])

##### AGE #####
train2['Age'].plot.hist()
plt.savefig('age_simple_hist.png')


def plot_hist_for_continuous(df_train, group_names, predictor_label, target_label, colors):
    """
    Takes two columns from train df and plots histogram
    :param df_train: data frame with train data
    :param group_names: names for 2 groups of target variable, first for false then for true values
    :param predictor_label: name of predictor variable
    :param target_label: name of target variable
    :param colors: list of colors for each group
    :return: saves a histogram with data distribution comparing target and predictor
    """
    zeroes = df_train[df_train[target_label] == False]
    ones = df_train[df_train[target_label] == True]
    zeroes[predictor_label].plot.hist(alpha=0.5, color=colors[0])
    ones[predictor_label].plot.hist(alpha=0.5, color=colors[1])
    plt.legend(group_names)
    plt.savefig(predictor_label + '_hist.png')


plot_hist_for_continuous(train2, ['above 50k', 'below 50k'], 'Age', 'Income.Group', ['green', 'orange'])

##### WORKCLASS #####
train2['Workclass'].value_counts() / train2.shape[0]


def plot_pivot_bar(df_train, predictor_label, target_label):
    predictor_pivot = df_train.pivot_table(index=predictor_label, values=target_label)
    predictor_pivot.plot.bar(figsize=(10, 12))
    plt.savefig(predictor_label + '_pivot.png')


plot_pivot_bar(train2, 'Workclass', 'Income.Group')

"""
# For the same effect as pivot - cross-tab and plot (add percent)
workclass_ct = pd.crosstab(train2['Workclass'], train2['Income.Group'], margins=True)
workclass_ct.iloc[:-1,:-1].plot.bar(figsize=(10, 12), stacked=True, color=['green', 'orange'], grid=False)
plt.savefig('workclass_incomegroup_cross_bar.png')

def percConvert(ser):
    return ser/float(ser[-1])

workclass_ct_perc = workclass_ct.apply(percConvert, axis=1)
workclass_ct_perc.iloc[:-1,:-1].plot.bar(figsize=(10, 12), stacked=True, color=['blue', 'pink'], grid=False)
plt.savefig('workclass_incomegroup_cross_bar_perc.png')
"""

##### EDUCATION #####
train2['Education'].value_counts() / train2.shape[0]
plot_pivot_bar(train2, 'Education', 'Income.Group')

##### MARTIAL STATUS #####
train2['Marital.Status'].value_counts() / train2.shape[0]
plot_pivot_bar(train2, 'Marital.Status', 'Income.Group')

##### OCCUPATION #####
train2['Occupation'].value_counts() / train2.shape[0]
plot_pivot_bar(train2, 'Occupation', 'Income.Group')

##### RELATIONSHIP #####
train2['Relationship'].value_counts() / train2.shape[0]
plot_pivot_bar(train2, 'Relationship', 'Income.Group')

##### RACE #####
train2['Race'].value_counts() / train2.shape[0]
plot_pivot_bar(train2, 'Race', 'Income.Group')

##### SEX #####
train2['Sex'].value_counts() / train2.shape[0]
plot_pivot_bar(train2, 'Sex', 'Income.Group')

##### HOURS PER WEEK #####
train2['Hours.Per.Week'].plot.hist()
plt.savefig('hours_simple_hist.png')

plot_hist_for_continuous(train2, ['above 50k', 'below 50k'], 'Hours.Per.Week', 'Income.Group', ['green', 'orange'])

##### NATIVE COUNTRY #####
train2['Native.Country'].value_counts() / train2.shape[0]
plot_pivot_bar(train2, 'Native.Country', 'Income.Group')


##### HANDLING NANS #####
def compare_nans(df_train, variable_labels):
    """
    Takes two variables from df and checks if the nans relate to the same records in both variables
    :param df_train: data frame with train data
    :param variable_labels: list of variables labels which have NaN values
    :return: a list with indexes of NaN which are in one variable, but not in the other
    """
    l0 = list(np.where(df_train[variable_labels[0]].isnull())[0])
    l1 = list(np.where(df_train[variable_labels[1]].isnull())[0])
    diff = list(set(l0) - set(l1))
    return diff


compare_nans(train2, ['Workclass', 'Occupation'])

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
train_clean = train2.copy()
test_clean = test2.copy()


# NaNs in Workclass, Occupation, Native.Country

def replace_nans(df_train, df_test, variable_labels, new_value):
    """
    Takes data frames with train and test values and replaces NaNs with new values
    :param df_train: data frame with train data
    :param df_test: data frame with test data
    :param variable_labels: list of variables labels which have NaN values
    :param new_value: new string values for NaN
    :return: two data frames with replaced NaNs
    """
    for var in variable_labels:
        df_train[var] = df_train[var].replace(np.nan, new_value, regex=True)
        df_test[var] = df_test[var].replace(np.nan, new_value, regex=True)


replace_nans(train_clean, test_clean, ['Workclass', 'Occupation', 'Native.Country'], 'Missing')


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
    new_label = variable_label + '_cat'
    df_train[new_label] = pd.cut(df_train[variable_label], cut_points, labels=labels)
    df_test[new_label] = pd.cut(df_test[variable_label], cut_points, labels=labels)


put_into_bins(train_clean, test_clean, 'Age', [0, 30, 45, 60, 100], ['Young', 'Middle Lower', 'Middle Upper', 'Senior'])
plot_pivot_bar(train_clean, 'Age_cat', 'Income.Group')


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


group_with_dict(train_clean, test_clean, 'Workclass', {'Private': 'private', 'Missing': 'missing',
                                                       'Self-emp-not-inc': 'self_employ', 'Self-emp-inc': 'self_employ',
                                                       'Federal-gov': 'gov', 'State-gov': 'gov',
                                                       'Local-gov': 'gov', 'Never-worked': 'no_pay',
                                                       'Without-pay': 'no_pay'})
plot_pivot_bar(train_clean, 'Workclass', 'Income.Group')

# Education - new categories
group_with_dict(train_clean, test_clean, 'Education', {'HS-grad': 'college', 'Some-college': 'college',
                                                       'Bachelors': 'degree', 'Masters': 'degree', 'Assoc-voc': 'docs',
                                                       '11th': 'college', 'Assoc-acdm': 'docs', '10th': 'college',
                                                       '7th-8th': 'secondary', 'Prof-school': 'docs',
                                                       '9th': 'secondary',
                                                       '12th': 'college', 'Doctorate': 'docs', '5th-6th': 'primary',
                                                       '1st-4th': 'primary', 'Preschool': 'primary'})
plot_pivot_bar(train_clean, 'Education', 'Income.Group')

# Marital.Status - new categories:
group_with_dict(train_clean, test_clean, 'Marital.Status', {'Married-civ-spouse': 'family', 'Never-married': 'single',
                                                            'Divorced': 'single', 'Separated': 'single',
                                                            'Widowed': 'single', 'Married-spouse-absent': 'single',
                                                            'Married-AF-spouse': 'family'})
plot_pivot_bar(train_clean, 'Marital.Status', 'Income.Group')


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
    cat_freq = df_train[variable_label].value_counts() / df_train.shape[0]
    cat_to_combine_index = cat_freq.loc[cat_freq.values < threshold].index
    for category in cat_to_combine_index:
        df_train[variable_label].replace({category: 'others'}, inplace=True)
        df_test[variable_label].replace({category: 'others'}, inplace=True)


group_minors(train_clean, test_clean, 'Occupation', 0.05)
plot_pivot_bar(train_clean, 'Occupation', 'Income.Group')

# Race - Put the last 3 in one group 'Other'
group_with_dict(train_clean, test_clean, 'Race', {'White': 'white', 'Black': 'black', 'Asian-Pac-Islander': 'other',
                                                  'Amer-Indian-Eskimo': 'other', 'Other': 'other'})
plot_pivot_bar(train_clean, 'Race', 'Income.Group')

# Hours.Per.Week - Most people work around 40 hours.Make bins: below 40, equals 40 and above 40
put_into_bins(train_clean, test_clean, 'Hours.Per.Week', [0, 39, 40, 100], ['below_mode', 'equals_mode', 'above_mode'])
plot_pivot_bar(train_clean, 'Hours.Per.Week_cat', 'Income.Group')

# Native.Country - Try making two categories: USA and Others
group_minors(train_clean, test_clean, 'Native.Country', 0.02)
plot_pivot_bar(train_clean, 'Native.Country', 'Income.Group')


########## 5. Predictive Modeling ###

# create dummies for categorical values

def create_dummies(df, variable_label):
    """
    Takes a df column with variable_label and creates dummies for it
    :param df: data frame name
    :param variable_label: name of column
    :return: df with added columns for dummies
    """
    dummies = pd.get_dummies(df[variable_label], prefix=variable_label)
    return pd.concat([df, dummies], axis=1)


def apply_dummies(df_train, df_test, variable_labels):
    """
    Takes data frames with train and test values and applies create_dummies function to given columns
    :param df_train: data frame with train data
    :param df_test: data frame with test data
    :param variable_labels: list of variables labels which need dummies columns
    :return:
    """
    for variable in variable_labels:
        df_train = create_dummies(df_train, variable)
        df_test = create_dummies(df_test, variable)
    return df_train, df_test


train_clean, test_clean = apply_dummies(train_clean, test_clean, ['Workclass', 'Education', 'Marital.Status',
                                                                  'Occupation', 'Race', 'Sex', 'Age_cat',
                                                                  'Hours.Per.Week_cat'])

########## 5. Predictive Modeling ###

# train and test split
holdout = test_clean
columns_for_modeling = list(set(list(train_clean.columns)) - {'ID', 'Age', 'Workclass', 'Education', 'Marital.Status',
                                                              'Occupation', 'Relationship', 'Race', 'Sex',
                                                              'Hours.Per.Week',
                                                              'Native.Country', 'Income.Group', 'Age_cat',
                                                              'Hours.Per.Week_cat'})
all_x = train_clean[columns_for_modeling]
all_y = train_clean['Income.Group']
train_x, test_x, train_y, test_y = train_test_split(all_x, all_y, test_size=0.2, random_state=0)

# modeling
log_reg = LogisticRegression()
dec_tree = DecisionTreeClassifier()
svc = SVC()
lin_svc = LinearSVC()
gauss = GaussianNB()
knn = KNeighborsClassifier(n_neighbors=3)
rand_forest = RandomForestClassifier(n_estimators=100)
gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

# Cross-Validation

def get_cross_val_score_table(algo_dict):
    """
    Takes list of algorithms and their estimators and creates accuracy table
    :param algo_dict: dictionary with algo names keys and algo estimator values
    :return: data frame with standard and cross validation score for every algorithm
    """
    models_summary = pd.DataFrame(index= range(len(algo_dict)), columns=['name', 'score', 'cross_val_mean_score', 'std_dev'])
    algo_names = list(algo_dict.keys())
    algo_est = list(algo_dict.values())
    for i in range(len(algo_dict)):
        models_summary.loc[i]['name'] = algo_names[i]
        est = algo_est[i]
        est.fit(train_x, train_y)
        models_summary.loc[i]['score'] = round(est.score(train_x, train_y) * 100, 2)
        models_summary.loc[i]['cross_val_mean_score'] = cross_val_score(est, all_x, all_y, cv=10).mean()
        models_summary.loc[i]['std_dev'] = cross_val_score(est, all_x, all_y, cv=10).std()
        models_summary.sort_values(by='cross_val_mean_score', ascending=False, inplace=True)
    return models_summary

algorithms = {'Logistic Regression': log_reg, 'Decision Tree': dec_tree, 'Support Vector Machine': svc,
              'Linear SVC': lin_svc, 'Naive Bayes': gauss, 'kNN': knn, 'Random Forest': rand_forest,
              'Gradient Boosting Algorithm': gbm}

models_accuracy_table = get_cross_val_score_table(algorithms)

# print coefficients for the best algorithm and predict
print('Coefficient: \n', lin_svc.coef_)
print('Intercept: \n', lin_svc.intercept_)
y_pred = lin_svc.predict(test_x)

coefficients = lin_svc.coef_
coefficients = coefficients.tolist()
flat_list = [item for sublist in coefficients for item in sublist]
coefficients_dict = dict(zip(columns_for_modeling, flat_list))

"""
Most relevant features:
- 'Age_cat_Young': 0.35
- 'Education_degree': -0.36
- 'Education_docs': -0.31
- 'Education_primary': 0.48
- 'Education_secondary': 0.43
- 'Marital.Status_single': 0.49
- 'Occupation_Other-service': 0.33
- 'Workclass_no_pay': 0.43
The highest probability to be below threshold is among young, single, 
not specified occupation, education primary or secondary.
"""