import pandas as pd 
import numpy  as np
import seaborn as sb
import matplotlib.pyplot as plt
import seaborn as sns
#Importing Data
affairs = pd.read_csv("D:/Training/ExcelR_2/Logistic_Regression/Affairs/affairs.csv")
affairs = affairs.drop(affairs.columns[0], axis =1)

#Creating a seperate dataframe which has only categorical variables
affairs_cat = affairs.select_dtypes(include = 'object').copy()

#Creating a seperate dataframe which has only continuous variables
affairs_conti = affairs.select_dtypes(exclude ='object').copy()

#data types
affairs.dtypes
affairs.columns

#Preserving the original dataframe in case of future use
affairs_ori = affairs

affairs.head(4)

affairs_cat.head(4)

##########################################################
###Exploratory Data Analysis for categorical variables

descriptive = affairs_cat.describe()

###Exploratory Data Analysis for continous variables
descriptive_conti = affairs_conti.describe()


#Looking at the different values of distinct categories in our variable.

affairs_cat['gender'].unique()
affairs_cat['children'].unique()



#No of unique categories 

len(affairs_cat['gender'].unique())
len(affairs_cat['children'].unique())


#Counting no of unique categories without any missing values

affairs_cat['gender'].nunique() 
affairs_cat['children'].nunique()


# No of missing values

affairs_cat['gender'].isnull().sum()
affairs_cat['children'].isnull().sum()

##Count plot / Bar Plot

sns.countplot(data = affairs_cat, x = 'gender')
sns.countplot(data = affairs_cat, x = 'children')


len(affairs_cat.columns)


#Exploratory data analysis for continous variables

# Correlation matrix 
affairs_conti.corr()


# getting boxplot of price with respect to each category of gears 

heat1 = affairs_conti.corr()
sns.heatmap(heat1, xticklabels=affairs_conti.columns, yticklabels=affairs_conti.columns, annot=True)


# Scatter plot between the variables along with histograms
sns.pairplot(affairs_conti)

# usage lambda and apply function
# apply function => we use to apply custom function operation on 
# each column
# lambda just an another syntax to apply a function on each value 
# without using for loop 
affairs.isnull().sum()


from tabulate import tabulate as tb
print(tb(descriptive,affairs.columns))

affairs.apply(lambda x:x.mean()) 
affairs.mean()

affairs.dtypes
affairs.columns



# Converting the variable affaris from continous to categorical 
affairs.dtypes
affairs_cat_cat = pd.cut(affairs.affairs,bins=[-1,0,50],labels=['No','Yes'])
affairs = affairs.drop(['affairs'],axis=1)
affairs = pd.concat([affairs,affairs_cat_cat],axis=1)


###### Creating dummy variables for the categorical data 

job_dum = pd.get_dummies(affairs.gender,drop_first = True)

df_dummies = pd.get_dummies(affairs, columns = ['affairs','gender', 'children'], drop_first = True)
affairs = df_dummies


                              
# Getting the barplot for the categorical columns (df[df.columns[0:30]])

sb.countplot(x="affairs_Yes",data= affairs,palette="hls")
pd.crosstab(affairs.affairs_Yes,affairs.gender_male).plot(kind="bar")


# Checking if we have na values or not 
affairs.isnull().sum() # No null values


#Model building 

import statsmodels.formula.api as sm
logit_model = sm.logit('affairs_Yes~age+yearsmarried+religiousness+education+occupation+rating+gender_male+children_yes',data = affairs).fit()


#summary
logit_model.summary()
y_pred = logit_model.predict(affairs)

affairs["pred_prob"] = y_pred
# Creating new column for storing predicted class of Attorney

# filling all the cells with zeroes
affairs["Att_val"] = np.zeros(601)

# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
affairs.loc[y_pred>=0.5,"Att_val"] = 1
affairs.Att_val

from sklearn.metrics import classification_report
classification_report(affairs.Att_val,affairs.affairs_Yes)

# confusion matrix 
confusion_matrix = pd.crosstab(affairs['affairs_Yes'],affairs.Att_val)

confusion_matrix
accuracy = (435+25)/(601) # 76.53
accuracy

# ROC curve 
from sklearn import metrics
# fpr => false positive rate
# tpr => true positive rate
fpr, tpr, threshold = metrics.roc_curve(affairs.affairs_Yes, y_pred)


# the above function is applicable for binary classification class 

plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive")
 
roc_auc = metrics.auc(fpr, tpr) # area under ROC curve 


### Dividing data into train and test data sets
affairs.drop("Att_val",axis=1,inplace=True)
from sklearn.model_selection import train_test_split

train,test = train_test_split(affairs,test_size=0.3)

# checking na values 
train.isnull().sum();test.isnull().sum()

# Building a model on train data set 

train_model = sm.logit('affairs_Yes~age+yearsmarried+religiousness+education+occupation+rating+gender_male+children_yes',data = train).fit()
train_model.summary()

#Dropping insignificant variables from the dataframe
train_model = sm.logit('affairs_Yes~age+yearsmarried+religiousness+occupation+rating+gender_male+children_yes',data = train).fit()
train_model.summary()

train_model = sm.logit('affairs_Yes~age+yearsmarried+religiousness+rating+gender_male',data = train).fit()
train_model.summary()

train_model = sm.logit('affairs_Yes~age+yearsmarried+religiousness+rating',data = train).fit()
train_model.summary()


#summary
train_model.summary()
train_pred = train_model.predict(train)

# Creating new column for storing predicted class of Attorney

# filling all the cells with zeroes
train["train_pred"] = np.zeros(420)

# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
train.loc[train_pred>0.5,"train_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(train['affairs_Yes'],train.train_pred)

confusion_matrix
accuracy_train = (300+27)/(420) 
accuracy_train

# Prediction on Test data set

test_pred = train_model.predict(test)

# Creating new column for storing predicted class of Attorney

# filling all the cells with zeroes
test["test_pred"] = np.zeros(181)

# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
test.loc[test_pred>0.5,"test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test['affairs_Yes'],test.test_pred)

confusion_matrix
accuracy_test = (123+8)/(181) 
accuracy_test

#ROC

# fpr => false positive rate
# tpr => true positive rate
fpr, tpr, threshold = metrics.roc_curve(test.affairs_Yes, test_pred)

plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive")
 
roc_auc = metrics.auc(fpr, tpr) # area under ROC curve 
