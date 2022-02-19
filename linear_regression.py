# To predict the satisfaction levels of employees using Linear Regression

## A linear regression model is fit using satisfaction level as dependent variable and the other variables as independent variables
X = hr_data[[ 'last_evaluation','number_project', 'average_montly_hours','time_spend_company', 'Work_accident', 'left', 'promotion_last_5years' , 'sales_coded', 'salary_coded']]
y = hr_data[['satisfaction_level']]

## Use 80-20 cross validation technique by splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state = 100 )

## Initialise a linear regression object and use the fit method to the train set
lm = LinearRegression()
lm.fit(X_train,y_train)

## Use the fitted model to predict satisfaction level on test set
y_pred = lm.predict(X_test)

## Compute performance metrics of the model with the actual and predicted test set labels
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
rmse = sqrt(mse)

