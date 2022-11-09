#!/usr/bin/env python
# coding: utf-8


#@ IMPORTING LIBRARIES AND DEPENDENCIES:
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# ## 1. Load the data

#@ dataset
df = pd.read_csv("C:/Users/yadwi/OneDrive/DataScience/MLZoomcamp/2022 Cohort/Mid term project/Data/archive/cirrhosis.csv")
df.head()


#@ Make data consistent by making columns names in lower cases and removing spaces
df.columns = df.columns.str.lower().str.replace(' ','_')
df.head()

#df.iloc[47].to_dict()


#@ checking data types of each variable in the dataframe
df.dtypes

df.info()

# ## 1.2 Explore the data

#@ Explore each columns of the dataset
for col in df.columns:
    print(col)
    print(df[col].unique()[:5]) # checking all unique values
    print(df[col].nunique()) # checking number of unique values
    print()


# #### Define the target variable
# #### Stage is the target variable

# #### Define categorical and numerical variables


#@ select all features with numerical and categorical data types seperately:
numerical = ['n_days','bilirubin','cholesterol','albumin','copper','alk_phos','sgot','tryglicerides','platelets','prothrombin']     
categorical = ['status', 'drug', 'sex', 'ascites', 'hepatomegaly', 'spiders','edema']

# #### Check for missing values

#@ check missing values
df.isnull().sum()

# #### Imputing missing values with mean and mode

#@ work on missingness
# imputing missingness in numerical variables by mean

for n in numerical:
    df[n].fillna(df[n].median(), inplace=True)

df.select_dtypes(include=(['int64', 'float64'])).isna().sum()


# imputing missingness in categorical variables by mode
for c in categorical:
    df[c].fillna(df[c].mode().values[0], inplace=True)

df.select_dtypes(include=(['object'])).isna().sum()

# #### Convert target variable into a binary variable

# Turning target variable into binary e.g., if Liver cirrhosis = 1 ,  No cirrhosis = 0
df.stage = np.where(df.stage == 4,1,0)
df.stage

# ## 1.3 EDA

df.stage.value_counts()

# correlation matrix for the numerical features of the dataset
pcorr = df[numerical].corr()

plt.figure(figsize = (16,5))
sns.heatmap(pcorr, 
            xticklabels=pcorr.columns,
            yticklabels=pcorr.columns,
            cmap='RdBu_r',
            annot=True)

# ## 1.4 Setting up the validation framework
# #### performing train/test/val split

df_train, df_test = train_test_split(df, test_size=0.2, random_state=11)

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = (df_train.status == 'stage').astype('int').values
y_test = (df_test.status == 'stage').astype('int').values

del df_train['stage']
del df_test['stage']


from sklearn.model_selection import train_test_split
# Split your data in train/val/test sets, with 60%/20%/20% distribution.
df_full_train, df_test = train_test_split(df, test_size=0.20, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

# Reset all Index
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
len(df_train), len(df_val), len(df_test)

y_train = df_train.stage.values
y_val = df_val.stage.values
y_test = df_test.stage.values

df_train_feat = df_train

del df_train['stage']
del df_val['stage']
del df_test['stage']

# ## 1.5 Feature Importance Analysis
# ### Mutual info score

def calculate_mi(series):
    return mutual_info_score(series, df_full_train.stage)

df_mi = df_full_train[categorical].apply(calculate_mi)
df_mi = df_mi.sort_values(ascending=False).to_frame(name='MI')



# * The mutual info score shows hepatomegaly, status, ascites, & Edema are some important features. Ascites is the main complication of cirrhosis. Thus, is relevant to to the stage of the liver cirrhosis. 
# 
# * Hepatomegaly is the condition of larger liver than normal. This is a critical indication of diseased liver. 
# 
# * Cirrhosis slows the normal flow of blood through the liver, thus increasing pressure in the vein that brings blood to the liver from the intestines and spleen. Swelling in the legs and abdomen. The increased pressure in the portal vein can cause fluid to accumulate in the legs (edema) and in the abdomen (ascites).
# 
# * Therefore, increased edema and ascites presence is indicator of presence of liver cirrhosis.

df_full_train[numerical].corrwith(df_full_train.stage).abs()

# #### One-hot encoding

# One-hot encoding
dv = DictVectorizer(sparse=False)

train_dict = df_train.to_dict(orient='records')
val_dict = df_val.to_dict(orient='records')
test_dict = df_test.to_dict(orient='records')

dv.fit(train_dict)

X_train = dv.transform(train_dict)
X_val = dv.transform(val_dict)
X_test = dv.transform(test_dict)


# ## 1.6 Training logistic regression model

model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# #### Accuracy with logistic regression

accuracy = model.score(X_val, y_val)
accuracy

# ## 1.7 Training decision tree classifier

dt = DecisionTreeClassifier(random_state = 1)
dt.fit(X_train, y_train)


y_pred = dt.predict_proba(X_val)[:,1]
roc_auc_score(y_val,y_pred)


# ## 1.7.1 Decision trees parameter tuning

# We will use  metric here and our aim would be to find the maximum AUC on applied on the validation set
depths = [1,2,3,4,5,6,7,8,9,10,11,12]

for depth in depths: # none means no restriction, this tree show grow as deep as possible
# we already know what happens when the tree keeps growing deeper
# but this none is for comparison

    dt = DecisionTreeClassifier(max_depth = depth)
    dt.fit(X_train, y_train)
    
    y_pred = dt.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, y_pred)
    
    print('%4s -> %.3f' % (depth,auc))

scores =[]
for d in[2,3,4]:
    for s in [1,2,5,10,15,20,100,200,500]:
        dt= DecisionTreeClassifier(max_depth=d,min_samples_leaf = s)
        dt.fit(X_train,y_train)
        
        y_pred = dt.predict_proba(X_val)[:,1]
        auc = roc_auc_score(y_val,y_pred)
        
        #print('(%4s , %3d) -> %.3f' % (d,s,auc))
        scores.append((d,s,auc))

# put this in df for easy visualisation
columns = ['max_depth','min_samples_leaf','auc']
df_scores =pd.DataFrame(scores, columns = columns)
df_scores.head()

# sorting auc values
df_scores.sort_values(by='auc',ascending=False)

# Pivot
df_scores_pivot = df_scores.pivot(index='min_samples_leaf',
                                 columns=['max_depth'], values =['auc'])
df_scores_pivot.round(3)

# visualize it using heatmap
sns.heatmap(df_scores_pivot, annot=True, fmt='.3f')

# the best max_depth values are somewhere in between 2 & 4
# try each of these values, try different min_samples_leaf values

scores =[]
for d in[2,3,4,5,6,10,15,20,None]:
    for s in [1,2,5,10,15,20,100,200,500]:
        dt= DecisionTreeClassifier(max_depth=d,min_samples_leaf = s)
        dt.fit(X_train,y_train)
        
        y_pred = dt.predict_proba(X_val)[:,1]
        auc = roc_auc_score(y_val,y_pred)
        
        #print('(%4s , %3d) -> %.3f' % (d,s,auc))
        scores.append((d,s,auc))

columns = ['max_depth','min_samples_leaf','auc']
df_scores =pd.DataFrame(scores, columns = columns)
df_scores.head()

# sorting auc values
df_scores.sort_values(by='auc',ascending=False)

# Pivot
df_scores_pivot = df_scores.pivot(index='min_samples_leaf',
                                 columns=['max_depth'], values =['auc'])
df_scores_pivot.round(3)

# visualize it using heatmap
sns.heatmap(df_scores_pivot, annot=True, fmt='.3f')

#### Final decision tree
# final decision tree
dt = DecisionTreeClassifier(max_depth = 4, min_samples_leaf = 15, random_state = 1)
dt.fit(X_train, y_train)


#### Decision tree final model on validation set
y_pred = dt.predict_proba(X_val)[:,1]
roc_auc_score(y_val, y_pred)


# ## 1.8 Training random forest classifier

# train the model
rf = RandomForestClassifier (n_estimators=10, random_state = 1) # here number of estimator is number of models we want to train
rf.fit(X_train,y_train)

# use trained model for prediction
y_pred = rf.predict_proba(X_val)[:,1]

auc_score = roc_auc_score(y_val, y_pred) # this is already pretty good without tuning

print(f'roc_auc_score with random forest classifier ={auc_score}')

# ## 1.8.1 Tuning the random forest classifier

# checking first row of validation set to see the probability
rf.predict_proba(X_val[[0]])

# increase the n_estimators or number of trees 
# iterate over many different values
scores =[]

for n in range(10,201,10):
    rf = RandomForestClassifier(n_estimators = n, random_state = 1)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, y_pred)
    scores.append((n,auc))

# check the scores in df
df_scores = pd.DataFrame(scores, columns=['n_estimators','auc'])
df_scores

#plot
plt.plot(df_scores.n_estimators, df_scores.auc)

scores = []

for d in [5,10,15]:
    for n in range(10,201,10):
        rf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=1)
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict_proba(X_val)[:,1]
        auc = roc_auc_score(y_val, y_pred)
        scores.append((d,n,auc))

columns = ['max_depth','n_estimators','auc']
df_scores = pd.DataFrame(scores, columns = columns)
df_scores.head()

for d in [5,10,15]:
    df_subset = df_scores[df_scores.max_depth == d]
    plt.plot(df_subset.n_estimators, df_subset.auc, label="max_depth %d" %d)
plt.legend()

max_depth = 15
scores = []

for s in [1,3,5,10,50]:
    for n in range (10,201,10):
        rf = RandomForestClassifier (n_estimators=n,
                                    max_depth= max_depth,
                                    min_samples_leaf=s,
                                    random_state=1)
        rf.fit(X_train,y_train)
        
        y_pred = rf.predict_proba(X_val)[:,1]
        auc = roc_auc_score(y_val, y_pred)
        scores.append((s,n,auc))

columns = ['min_samples_leaf','n_estimators','auc']
df_scores = pd.DataFrame(scores, columns = columns)
df_scores.head()

colors = ['black', 'blue', 'orange', 'red', 'grey']
values = [1, 3, 5, 10, 50]

for s, col in zip(values, colors):
    df_subset = df_scores[df_scores.min_samples_leaf == s]
    
    plt.plot(df_subset.n_estimators, df_subset.auc,
             color=col,
             label='min_samples_leaf=%d' % s)

plt.legend()  

rf = RandomForestClassifier(n_estimators=50,
                            max_depth=15,
                            min_samples_leaf=50,
                            random_state=1)
rf.fit(X_train, y_train)


# use trained model for prediction
y_pred = rf.predict_proba(X_val)[:,1]

roc_auc_score(y_val, y_pred) #


# ## 1.9 Final model
# 
# * Based on the best AUC, I will select the random forest model (rf) and will train this model on train and validation sets

df_full_train = df_full_train.reset_index(drop=True)
y_full_train = df_full_train.stage.values

del df_full_train['stage']


dicts_full_train = df_full_train.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)

# we have y_test, we only need feature matrix here
dicts_test = df_test.to_dict(orient='records')
X_test = dv.transform(dicts_test)

model_rf = RandomForestClassifier(n_estimators=50,max_depth=15,min_samples_leaf=50,random_state=1)

model_rf.fit(X_train, y_train)

y_pred = model.predict_proba(X_test)[:,1]
roc_auc_score(y_test, y_pred)

# ## 1.10 Save the model for deployment

model_output_file = f'model_rf.bin'

with open(model_output_file,'wb') as f_out:
     pickle.dump((dv,model_rf),f_out)


print(f'the model is saved to {model_output_file}')