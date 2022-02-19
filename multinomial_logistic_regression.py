# To classify salary of employees using Multinomial Logistic Regression.

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

## The salary column is a categorical column, with three levels – high, medium and low. A multinomial logistic regression is fit using all features to classify salary.
X = hr_data[['satisfaction_level',  'last_evaluation','number_project', 'average_montly_hours','time_spend_company', 'Work_accident', 'left', 'promotion_last_5years', 'sales_coded']]
y = hr_data.salary_coded

## Use 80-20 cross validation technique by splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state = 100 )

## Fit a multinomial logistic regression model on the train set
logistic_reg = LogisticRegression() 
logistic_reg.fit(X_train, y_train) 

## Using the fitted model to predict on test set
y_pred = logistic_reg.predict(X_test)

## Computing the accuracy of the classification
acc = accuracy_score(y_test, y_pred)
