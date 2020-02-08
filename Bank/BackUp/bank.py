import pandas as pd 
import numpy  as np
import seaborn as sb
import matplotlib.pyplot as plt
import seaborn as sns
#Importing Data
bank = pd.read_csv("D:/Training/ExcelR_2/Logistic_Regression/Bank/bank-full.csv", sep=";")

#Creating a seperate dataframe which has only categorical variables
bank_cat = bank.select_dtypes(include = 'object').copy()

#Creating a seperate dataframe which has only continuous variables
bank_conti = bank.select_dtypes(include ='int64').copy()

#data types
bank.dtypes
bank.columns

# Preserving the original dataframe in case of future use
bank_ori = bank

bank.head(4)

bank_cat.head(4)

##########################################################
###Exploratory Data Analysis for categorical variables

descriptive = bank_cat.describe()

# the following is the data of the people who have subscribed to the term deposite
des_y_yes = bank_cat.loc[bank_cat['y'] == "yes"]

descriptive_yes = des_y_yes.describe()

# the following is the data of the people who have not subscribed to the term deposite
des_n_yes = bank_cat.loc[bank_cat['y'] == "no"]

descriptive_no = des_n_yes.describe()



#Looking at the different values of distinct categories in our variable.

bank_cat['job'].unique()
bank_cat['marital'].unique()
bank_cat['education'].unique()
bank_cat['default'].unique()
bank_cat['housing'].unique()
bank_cat['loan'].unique()
bank_cat['contact'].unique()
bank_cat['month'].unique()
bank_cat['poutcome'].unique()
bank_cat['y'].unique()


#No of unique categories 

len(bank_cat['job'].unique())
len(bank_cat['marital'].unique())
len(bank_cat['education'].unique())
len(bank_cat['default'].unique())
len(bank_cat['housing'].unique())
len(bank_cat['loan'].unique())
len(bank_cat['contact'].unique())
len(bank_cat['month'].unique())
len(bank_cat['poutcome'].unique())
len(bank_cat['y'].unique())

#Counting no of unique categories without any missing values

bank_cat['job'].nunique() 
bank_cat['marital'].nunique()
bank_cat['education'].nunique()
bank_cat['default'].nunique()
bank_cat['housing'].nunique()
bank_cat['loan'].nunique()
bank_cat['contact'].nunique()
bank_cat['month'].nunique()
bank_cat['poutcome'].nunique()
bank_cat['y'].nunique()

# No of missing values

bank_cat['job'].isnull().sum()
bank_cat['marital'].isnull().sum()
bank_cat['education'].isnull().sum()
bank_cat['default'].isnull().sum()
bank_cat['housing'].isnull().sum()
bank_cat['loan'].isnull().sum()
bank_cat['contact'].isnull().sum()
bank_cat['month'].isnull().sum()
bank_cat['poutcome'].isnull().sum()
bank_cat['y'].isnull().sum()

##Count plot / Bar Plot

sns.countplot(data = bank_cat, x = 'job')
sns.countplot(data = bank_cat, x = 'marital')
sns.countplot(data = bank_cat, x = 'education')
sns.countplot(data = bank_cat, x = 'default')
sns.countplot(data = bank_cat, x = 'housing')
sns.countplot(data = bank_cat, x = 'loan')
sns.countplot(data = bank_cat, x = 'contact')
sns.countplot(data = bank_cat, x = 'month')
sns.countplot(data = bank_cat, x = 'poutcome')
sns.countplot(data = bank_cat, x = 'y')

len(bank_cat.columns)


#Box Plot

sns.boxplot(data = bank_cat, x='job', y='y')

#Exploratory data analysis for continous variables

# Correlation matrix 
bank_conti.corr()


# getting boxplot of price with respect to each category of gears 
sns.boxplot(x="Age_08_04",y="Price",data=bank_conti)

heat1 = bank_conti.corr()
sns.heatmap(heat1, xticklabels=bank_conti.columns, yticklabels=bank_conti.columns, annot=True)


# Scatter plot between the variables along with histograms
sns.pairplot(bank_conti)

# usage lambda and apply function
# apply function => we use to apply custom function operation on 
# each column
# lambda just an another syntax to apply a function on each value 
# without using for loop 
bank.isnull().sum()


from tabulate import tabulate as tb
print(tb(descriptive,bank.columns))

bank.apply(lambda x:x.mean()) 
bank.mean()

bank.dtypes
bank.columns

job_dum = pd.get_dummies(bank.job,drop_first = True)

df_dummies = pd.get_dummies(bank, columns = ['job', 'marital', 'education','default','housing','loan','contact','month','poutcome','y'], drop_first = True)
bank = df_dummies
bank_total_dum = pd.get_dummies(bank_ori, columns = ['job', 'marital', 'education','default','housing','loan','contact','month','poutcome','y'], drop_first = False)


#Removing special characters from the dataframe
#####rankings_pd.rename(columns = {'test':'TEST'}, inplace = True) ### Code for reference 

bank.columns=bank.columns.str.replace('-','_')
bank.columns

#Normalizing data of age, balance, day, duration, compaign, pdays & previous

from sklearn import preprocessing

age = bank[['age']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
age_scaled = min_max_scaler.fit_transform(age)
age = pd.DataFrame(age_scaled)
age.columns = ['age']
age 


balance = bank[['balance']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
balance_scaled = min_max_scaler.fit_transform(balance)
balance = pd.DataFrame(balance_scaled)
balance.columns = ['balance']
balance 

day = bank[['day']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
day_scaled = min_max_scaler.fit_transform(day)
day = pd.DataFrame(day_scaled)
day.columns = ['day']
day 

duration = bank[['duration']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
duration_scaled = min_max_scaler.fit_transform(duration)
duration = pd.DataFrame(duration_scaled)
duration.columns = ['duration']
duration 

campaign= bank[['campaign']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
campaign_scaled = min_max_scaler.fit_transform(campaign)
campaign = pd.DataFrame(campaign_scaled)
campaign.columns = ['compaign']
campaign 

pdays= bank[['pdays']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
pdays_scaled = min_max_scaler.fit_transform(pdays)
pdays = pd.DataFrame(pdays_scaled)
pdays.columns = ['pdays']
pdays 

previous= bank[['previous']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
previous_scaled = min_max_scaler.fit_transform(previous)
previous = pd.DataFrame(previous_scaled)
previous.columns = ['previous']
previous 


trans_data = [age,balance,day,duration,campaign,pdays,previous]
transformed = pd.concat(trans_data)

## Dropping categorical variables and replacing them with normalized variables
bank_conti

bank = bank.drop(['age','balance','day','duration','campaign','pdays','previous'], axis = 1)                       
bank_normal = pd.concat(['age','balance','day','duration','campaign','pdays','previous']), axis = 1)   

bank = 
                              
## df_dummies = pd.get_dummies(df[df.columns[0:30]]), drop_first = True
# Getting the barplot for the categorical columns (df[df.columns[0:30]])

sb.countplot(x="job_blue_collar",data= bank,palette="hls")
pd.crosstab(bank.job_entrepreneur,bank.marital_single).plot(kind="bar")

#Imputating the missing values with most repeated values in that column  

sns.catplot(x="job_blue_collar", y="education_secondary", hue="marital_married", kind="swarm", data=bank);

# lambda x:x.fillna(x.value_counts().index[0]) 


# Checking if we have na values or not 
bank.isnull().sum() # No null values


#Model building 

import statsmodels.formula.api as sm
logit_model = sm.logit('y_yes~age+balance+day+duration+campaign+pdays+previous+job_blue_collar+job_entrepreneur+job_housemaid+job_management+job_retired+job_self_employed+job_services+job_student+job_technician+job_unemployed+job_unknown+marital_married+marital_single+education_secondary+education_tertiary+education_unknown+default_yes+housing_yes+loan_yes+contact_telephone+contact_unknown+month_aug+month_dec+month_feb+month_jan+month_jul+month_jun+month_mar+month_may+month_nov+month_oct+month_sep+poutcome_other+poutcome_success+poutcome_unknown',data = bank).fit()


#summary
logit_model.summary()
y_pred = logit_model.predict(bank)

bank["pred_prob"] = y_pred
# Creating new column for storing predicted class of Attorney

# filling all the cells with zeroes
bank["Att_val"] = np.zeros(45211)

# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
bank.loc[y_pred>=0.5,"Att_val"] = 1
bank.Att_val

from sklearn.metrics import classification_report
classification_report(bank.Att_val,bank.y_yes)

# confusion matrix 
confusion_matrix = pd.crosstab(bank['y_yes'],bank.Att_val)

confusion_matrix
accuracy = (38940+982)/(45211) # 88.30
accuracy

# ROC curve 
from sklearn import metrics
# fpr => false positive rate
# tpr => true positive rate
fpr, tpr, threshold = metrics.roc_curve(bank.y_yes, y_pred)


# the above function is applicable for binary classification class 

plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive")
 
roc_auc = metrics.auc(fpr, tpr) # area under ROC curve 


### Dividing data into train and test data sets
bank.drop("Att_val",axis=1,inplace=True)
from sklearn.model_selection import train_test_split

train,test = train_test_split(bank,test_size=0.3)

# checking na values 
train.isnull().sum();test.isnull().sum()

# Building a model on train data set 

train_model = sm.logit('y_yes~age+balance+day+duration+campaign+pdays+previous+job_blue_collar+job_entrepreneur+job_housemaid+job_management+job_retired+job_self_employed+job_services+job_student+job_technician+job_unemployed+job_unknown+marital_married+marital_single+education_secondary+education_tertiary+education_unknown+default_yes+housing_yes+loan_yes+contact_telephone+contact_unknown+month_aug+month_dec+month_feb+month_jan+month_jul+month_jun+month_mar+month_may+month_nov+month_oct+month_sep+poutcome_other+poutcome_success+poutcome_unknown',data = train).fit()

#summary
train_model.summary()
train_pred = train_model.predict(train)

# Creating new column for storing predicted class of Attorney

# filling all the cells with zeroes
train["train_pred"] = np.zeros(31647)

# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
train.loc[train_pred>0.5,"train_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(train['y_yes'],train.train_pred)

confusion_matrix
accuracy_train = (27228+1294)/(31647) # 90.12
accuracy_train

# Prediction on Test data set

test_pred = train_model.predict(test)

# Creating new column for storing predicted class of Attorney

# filling all the cells with zeroes
test["test_pred"] = np.zeros(13564)

# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
test.loc[test_pred>0.5,"test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test['y_yes'],test.test_pred)

confusion_matrix
accuracy_test = (11703+161)/(13564) # 87.466
accuracy_test

