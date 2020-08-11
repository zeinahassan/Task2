# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 02:42:05 2020

@author: Zeina Hassan
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def describe_data(df): #Function to get info about the dataset
    print("Data Types:")
    print(df.dtypes)
    print("Rows and Columns:")
    print(df.shape)
    print("Column Names:")
    print(df.columns)
    print("Null Values:")
    print(df.apply(lambda x: sum(x.isnull()) / len(df)))

train=pd.read_csv('training.csv')
#describe_data(train) #Printing to know the datatypes and nulls

#Seperating numercial columns from categorical ones to work on them seperately
numeric_features = train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = train.select_dtypes(include=['object']).drop(['classLabel'], axis=1).columns

#Creating two pipielines for both features where there is a SimpleImputer to handle null values

#A scaler function is included to ensure that all the values in each column are on the same scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

#OneHotEncoder is used to transform each unique value in the categorical columns into a new column containing a 0 or 1 dependant on wether or not the value is present.    
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

#ColumnTransfomer to concatenate both the numeric and categorical transformers into an object called preprocessor.    
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

#Split the training data into a training and test set
x = train.drop('classLabel', axis=1)
y = train['classLabel']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#Creating a logistic regression model that performs the defined transformations before fitting or predicting.
model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(solver='lbfgs'))])
    
#Fitting the model on the training set, and obtaining an accuracy score using the built in score method    
model.fit(x_train, y_train)
print("Model score: %.2f" % model.score(x_test, y_test))

#Reading the validation file and uing our model to predict then calculating the accuracy score
valid=pd.read_csv('validation.csv')

x_valid = valid.drop('classLabel', axis=1)
y_valid = valid['classLabel']

predictions = model.predict(x_valid)
print("Accuracy score: %.2f" % accuracy_score(y_valid, predictions))

