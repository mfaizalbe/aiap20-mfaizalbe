#!/usr/bin/env python
# coding: utf-8

# ## For Business Understanding

# ### Import dataset

# In[1]:


import pandas as pd
import numpy as np
import sqlite3

# connect to your SQLite database file
conn = sqlite3.connect('data/bmarket.db')

# load table into a DataFrame
df = pd.read_sql_query("SELECT * FROM bank_marketing", conn)

# view the first few rows
print(df.head())

# close the connection
conn.close()


# ### 'Subscription Status' is the dependent variable as it represents the outcome we're predicting whether a client subscribes to a term deposit.

# ## For Data Understanding 

# ### Descriptive statistics

# In[2]:


# get descriptive statistics
df.describe(include='all')


# In[3]:


# get dataframe info such as data type and if there are missing values
df.info()


# ### The dataset has missing values in Housing Loan (24,789) and Personal Loan (4,146).

# ## Data cleaning

# In[4]:


# remove duplicates
df = df.drop_duplicates()
df.shape


# In[5]:


# unique values for each column
for column in df.columns:
    print(f"Unique values in '{column}':")
    print(df[column].unique())
    print("-" * 50)


# In[6]:


# drop 'Client ID' identifier
df = df.drop('Client ID', axis=1)

# remove ' years' text from the 'Age' column
df['Age'] = df['Age'].astype(str).str.replace(' years', '', regex=False)

# convert the 'Age' column to numeric (force errors to NaN for invalid values)
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

# remove age 150 years (assuming they are outliers)
df = df[df['Age'] <= 149]

# reclassifying 'Occupation' into broader meaningful categories for better analysis and modeling
occupation_map = {
    'unknown': 'others',
    'blue-collar': 'blue_collar',
    'technician': 'blue_collar',
    'admin.': 'white_collar',
    'services': 'white_collar', 
    'management': 'white_collar',
    'housemaid': 'others',
    'retired': 'others',
    'unemployed': 'others',
    'entrepreneur': 'entrepreneur',
    'self-employed': 'entrepreneur',
    'student': 'student'
}

# apply updated mapping for 'Occupation'
df['Occupation'] = df['Occupation'].replace(occupation_map)
    
# replace 'None' with 'unknown' in 'Marital Status'
df['Marital Status'].fillna('unknown', inplace=True)  
    
# reclassifying 'Education' into broader meaningful categories for better analysis and modeling
education_mapping = {
    'high.school': 'secondary',
    'basic.9y': 'secondary',
    'basic.4y': 'primary',
    'basic.6y': 'primary',
    'professional.course': 'vocational',
    'university.degree': 'university',
    'illiterate': 'no_education',
    'unknown': 'unknown'
}

# apply the mapping to the 'Education Level' column
df['Education Level'] = df['Education Level'].map(education_mapping)   

# replace 'None' with 'unknown' in 'Personal Loan' and 'Housing Loan'
df['Housing Loan'].fillna('unknown', inplace=True)
df['Personal Loan'].fillna('unknown', inplace=True)

# replace 'Cell' with 'cellular' and 'Telephone' with 'telephone' in the 'Contact Method' column
df['Contact Method'].replace({'Cell': 'cellular', 'Telephone': 'telephone'}, inplace=True)

# verify changes
for column in df.columns:
    print(f"Unique values in '{column}':")
    print(df[column].unique())
    print("-" * 50)
print("\n")
df.info()


# ## Visualization between independent and dependent variables

# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# create Age Bands
age_bins = [18, 29, 39, 49, 59, 69, 100]
age_labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70+']
df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

# create bands for 'Previous Contact Days'
contact_days_bins = [-1, 5, 10, 20, 30, 60]
contact_days_labels = ['0-5', '6-10', '11-20', '21-30', '31+']
df['Contact Days Group'] = pd.cut(df['Previous Contact Days'], bins=contact_days_bins, labels=contact_days_labels, right=False)

# create bands for 'Campaign Calls' (assuming negative numbers for failed attempts)
bins = [-float('inf'), 0, 5, 10, 20, 30, float('inf')]
labels = ['Failed Attempts', '0 to 5', '5 to 10', '10 to 20', '20 to 30', '30+'] 
df['Campaign Calls Binned'] = pd.cut(df['Campaign Calls'], bins=bins, labels=labels)

# count the occurrences of each subscription status
subscription_counts = df['Subscription Status'].value_counts()

# plotting the subscription status counts
plt.figure(figsize=(8, 6))
sns.barplot(x=subscription_counts.index, y=subscription_counts.values)
plt.xlabel('Subscription Status')
plt.ylabel('Count')
plt.title('Number of Subscription Status (Yes vs. No)')
plt.show()

# plot 'Age' group vs 'Subscription Status'
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Age Group', hue='Subscription Status')
plt.title('Subscription Status by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# plot 'Occupation' vs 'Subscription Status'
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Occupation', hue='Subscription Status')
plt.title('Subscription Status by Occupation')
plt.xlabel('Occupation')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# plot 'Education Level' vs 'Subscription Status'
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Education Level', hue='Subscription Status')
plt.title('Subscription Status by Education Level')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# plot 'Personal Loan' vs 'Subscription Status'
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Personal Loan', hue='Subscription Status')
plt.title('Subscription Status by Personal Loan')
plt.xlabel('Personal Loan')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# plot 'Housing Loan' vs 'Subscription Status'
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Housing Loan', hue='Subscription Status')
plt.title('Subscription Status by Housing Loan')
plt.xlabel('Housing Loan')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# plot 'Contact Method' vs 'Subscription Status'
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Contact Method', hue='Subscription Status')
plt.title('Subscription Status by Contact Method')
plt.xlabel('Contact Method')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# plot 'Previous Contact Days' group vs 'Subscription Status'
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Contact Days Group', hue='Subscription Status')
plt.title('Subscription Status by Contact Days Group')
plt.xlabel('Previous Contact Days')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# plot 'Campaign Calls' group vs 'Subscription Status'
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Campaign Calls Binned', hue='Subscription Status',)
plt.title('Campaign Calls vs Subscription Status')
plt.xlabel('Campaign Calls')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# plot 'Credit Default' vs 'Subscription Status'
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Credit Default', hue='Subscription Status')
plt.title('Subscription Status by Credit Default')
plt.xlabel('Credit Default')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# drop binned columns
df = df.drop(columns=[col for col in ['Age Group', 'Contact Days Group', 'Campaign Calls Binned'] if col in df.columns])

# create a label encoder
encoder = LabelEncoder()

# list of categorical columns to encode
categorical_cols = ['Occupation', 'Marital Status', 'Education Level', 'Credit Default', 
                    'Housing Loan', 'Personal Loan', 'Contact Method']

# apply label encoding to categorical columns
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col].astype(str))

# define 'X' as the features (all columns except 'Subscription Status')
X = df.drop(['Subscription Status'], axis=1)

# compute the correlation matrix of the features (X)
correlation_matrix = X.corr()

# plot the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix of Features")
plt.show()


# ## For Classification Task

# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc, roc_auc_score

# initialize the label encoder
encoder = LabelEncoder()

# columns to encode
categorical_cols = ['Occupation', 'Marital Status', 'Education Level', 'Credit Default', 
                    'Housing Loan', 'Personal Loan', 'Contact Method', 'Subscription Status']

for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col].astype(str))

# split data into features and target
X = df.drop('Subscription Status', axis=1)  # features (excluding target column)
y = df['Subscription Status']  # target variable

# stratified sampling to ensure similar distribution of the target in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ### We use stratified sampling here because it is an imbalanced data.

# ### Logistic Regression

# In[9]:


# train the logistic regression model
logreg_model = LogisticRegression(solver='liblinear', random_state=42)
logreg_model.fit(X_train, y_train)

# evaluate model
logreg_accuracy = accuracy_score(y_test, logreg_model.predict(X_test))
print(f"Accuracy of Logistic Regression: {logreg_accuracy}")
print("Classification Report for Logistic Regression:")
print(classification_report(y_test, logreg_model.predict(X_test)))

# calculate ROC curve and AUC in one line
fpr, tpr, _ = roc_curve(y_test, logreg_model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

# plot ROC curve
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # diagonal line (no-discrimination line)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Logistic Regression')
plt.legend(loc='lower right')
plt.show()


# ### Decision Tree Classifier

# In[10]:


# train the decision tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# evaluate model
dt_accuracy = accuracy_score(y_test, dt_model.predict(X_test))
print(f"Accuracy of Decision Tree: {dt_accuracy}")
print("Classification Report for Decision Tree:")
print(classification_report(y_test, dt_model.predict(X_test)))

# calculate ROC curve and AUC in one line
fpr, tpr, _ = roc_curve(y_test, dt_model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

# plot ROC curve
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # diagonal line (no-discrimination line)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Decision Tree Classifier')
plt.legend(loc='lower right')
plt.show()


# ### Random Forest Classifier

# In[11]:


# train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# evaluate model
rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
print(f"Accuracy of Random Forest: {rf_accuracy}")
print("Classification Report for Random Forest:")
print(classification_report(y_test, rf_model.predict(X_test)))

# calculate ROC curve and AUC in one line
fpr, tpr, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

# plot ROC curve
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # diagonal line (no-discrimination line)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest Classifier')
plt.legend(loc='lower right')
plt.show()


# ### Naive Bayes

# In[12]:


# initialize and train the Naive Bayes model
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train, y_train)

# make predictions
y_pred = naive_bayes_model.predict(X_test)

# evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Naive Bayes: {accuracy:.4f}")
print("Classification Report for Naive Bayes:")
print(classification_report(y_test, y_pred))

# calculate ROC curve and AUC in one line
fpr, tpr, _ = roc_curve(y_test, naive_bayes_model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

# plot ROC curve
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # diagonal line (no-discrimination line)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Naive Bayes')
plt.legend(loc='lower right')
plt.show()


# The results indicate that **Logistic Regression** shows the best performance with an accuracy of 89.55% and an ROC AUC of 0.72, though it struggles with identifying Yes subscribers. 
# 
# **Random Forest** performs slightly better than **Decision Tree** on the imbalanced dataset, with an accuracy of 87.39% and an ROC AUC of 0.67. Both models have difficulty with the minority Yes class subscribers. 
# 
# **Naive Bayes** delivers comparable performance with an accuracy of 89.43% and an ROC AUC of 0.73, but still struggles with recall for the Yes class. 
# 
# To improve results, we will try **XGBoost**.

# ### XGBoost

# In[13]:


import xgboost as xgb

# convert the data into DMatrix format for XGBoost (optional but efficient for large datasets)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# define the model parameters for XGBoost
params = {
    'objective': 'binary:logistic',  # binary classification
    'eval_metric': 'logloss',        # evaluation metric (log loss)
    'max_depth': 6,                  # max depth of the tree
    'eta': 0.1,                      # learning rate
    'subsample': 0.8,                # fraction of data used for each tree
    'colsample_bytree': 0.8,         # fraction of features used for each tree
    'scale_pos_weight': 1.5,         # balancing the class weights (important for imbalanced data)
}

# train the XGBoost model
model = xgb.train(params, dtrain, num_boost_round=100)

# make predictions
y_pred = model.predict(dtest)

# convert probabilities to binary outcomes (0 or 1)
y_pred_binary = [1 if i > 0.5 else 0 for i in y_pred]

# evaluate the model
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy of XGBoost: {accuracy}")
print("Classification Report for XGBoost:")
print(classification_report(y_test, y_pred_binary))

# calculate ROC-AUC score
roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC-AUC Score: {roc_auc:.2f}")

# plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # diagonal line (no-discrimination line)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for XGBoost')
plt.legend(loc='lower right')
plt.show()


# **Logistic Regression** is the best model with the highest accuracy (89.55%) and a ROC AUC of 0.72, but has low recall for the Yes class. 
# 
# **XGBoost** (accuracy: 89.31%, ROC AUC: 0.73) offers a slight improvement in AUC but still struggles with Yes class recall. 
# 
# **Random Forest** and **Decision Tree** perform worse, with ROC AUC scores of 0.67 and 0.60, respectively, and also show bias toward the majority class. 
# 
# **Naive Bayes** (accuracy: 89.43%, ROC AUC: 0.73) shows similar performance to **XGBoost**, with a slightly better accuracy and a comparable ROC AUC, but still faces challenges with the minority class recall. 
# 
# While **XGBoost** and **Naive Bayes** offer good performance, class balancing and data augmentation are crucial for improving recall for the Yes class.
