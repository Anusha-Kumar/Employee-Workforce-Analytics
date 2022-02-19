# DATA PREPROCESSING

## Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

## Reading dataset
hr_data = pd.read_csv("HR_comma_sep.csv")   #https://www.kaggle.com/jacksonchou/hr-analytics/data
hr_data.head()

hr_data.info()

## Checking for missing values
hr_data[hr_data.isnull().any(axis=1)] ## Output - No missing values

## Label encoding for categorical variables - Sales and Salary
label_encoder = LabelEncoder()
hr_data['sales_coded'] = label_encoder.fit_transform(hr_data['sales'])
hr_data['salary_coded'] = label_encoder.fit_transform(hr_data['salary'])

##Standardizing numerical data
hr_data[hr_data.columns[0:8]] = StandardScaler().fit_transform(hr_data[hr_data.columns[0:8]])
hr_data.head()
