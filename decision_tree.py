# To predict the satisfaction levels of employees using decision trees.

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from math import sqrt
from sklearn.tree import DecisionTreeRegressor  

## A decision tree regressor is implemented for predicting satisfaction level of employees
X = hr_data[[ 'last_evaluation','number_project', 'average_montly_hours','time_spend_company', 'Work_accident', 'left', 'promotion_last_5years' , 'sales_coded', 'salary_coded']]
y = hr_data[['satisfaction_level']]

## Use 80-20 cross validation technique by splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state = 100 )

## Fitting a linear regression model on the train set
regressor = DecisionTreeRegressor(random_state = 0)    
regressor.fit(X_train, y_train) 

## Using the fitted model to predit on the test set
y_pred = regressor.predict(X_test)

## Compute performance metrics of the model with the actual and predicted test set labels
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
rmse = sqrt(mse)
