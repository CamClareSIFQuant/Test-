''' CamSIF Quant Coding Task - 2024 Michaleamas 
    

'''

from IPython.display import clear_output
clear_output(wait=False)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import yfinance as yf
from sklearn.linear_model import LinearRegression

start = '2019-11-22'
end = '2023-05-20'
stock_market_data = yf.download("AAPL", start, end)
stock_market_data['Close'].plot(figsize=(10, 6))
plt.title("Apple Stock Price", fontsize=17)
plt.ylabel('Price', fontsize=14)
plt.xlabel('Time', fontsize=14)
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
plt.show()

def create_train_test_set(dataframe_stock_market_data):

  features = dataframe_stock_market_data.drop(columns=['Adj Close'], axis=1)
  target = dataframe_stock_market_data['Adj Close']

  data_length = dataframe_stock_market_data.shape[0]

  print("The historical stock market data's length is: ", str(data_length))

  train_split = int(data_length * 0.88)
  print("The training dataset's length is: ", str(train_split))

  val_split = train_split + int(data_length * 0.1)
  print("The validation set's length is: ", str(int(data_length * 0.1)))

  print("The test set's length: ", str(int(data_length * 0.02)))

  # Splitting features and target into train, validation and test samples 
  X_train, X_val, X_test = features[:train_split], features[train_split:val_split], features[val_split:]
  Y_train, Y_val, Y_test = target[:train_split], target[train_split:val_split], target[val_split:]

  #print shape of samples
  print("\nThe shape of samples on x axis (time):")
  print(X_train.shape, X_val.shape, X_test.shape)
  
  print("\nThe shape of samples on y axis (stock price):")
  print(Y_train.shape, Y_val.shape, Y_test.shape)
    
  return X_train, X_val, X_test, Y_train, Y_val, Y_test


# Generate datasets
X_train, X_validate, X_test, Y_train, Y_validate, Y_test = create_train_test_set(stock_market_data)

# Initialize the linear regression model
linear_regression_model = LinearRegression()

# train the model using training set
linear_regression_model.fit(X_train, Y_train)

print("Linear Regression Model's Coefficients: \n", linear_regression_model.coef_)
print("Linear Regression Model's Intercept: \n", linear_regression_model.intercept_)

print("The Performance of our model (R^2): ", linear_regression_model.score(X_train, Y_train))

def get_mape(y_truth, y_predicted): 
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_truth, y_predicted = np.array(y_truth), np.array(y_predicted)
    return np.mean(np.abs((y_truth - y_predicted) / y_truth)) * 100


# Get predicted data for training set, validation set and test set.
Y_train_predicted = linear_regression_model.predict(X_train)
Y_validate_predicted = linear_regression_model.predict(X_validate)
Y_test_predicted = linear_regression_model.predict(X_test)

# print out prediction result of these three datasets respectively
print("Training R-squared: ",round(metrics.r2_score(Y_train,Y_train_predicted),2))
print("Training Explained Variation: ",round(metrics.explained_variance_score(Y_train,Y_train_predicted),2))
print('Training MAPE:', round(get_mape(Y_train,Y_train_predicted), 2)) 
print('Training Mean Squared Error:', round(metrics.mean_squared_error(Y_train,Y_train_predicted), 2)) 
print("Training RMSE: ",round(np.sqrt(metrics.mean_squared_error(Y_train,Y_train_predicted)),2))
print("Training MAE: ",round(metrics.mean_absolute_error(Y_train,Y_train_predicted),2))

print('------------------------------------------------')

print("Validation R-squared: ",round(metrics.r2_score(Y_validate,Y_validate_predicted),2))
print("Validation Explained Variation: ",round(metrics.explained_variance_score(Y_validate,Y_validate_predicted),2))
print('Validation MAPE:', round(get_mape(Y_validate,Y_validate_predicted), 2)) 
print('Validation Mean Squared Error:', round(metrics.mean_squared_error(Y_train,Y_train_predicted), 2)) 
print("Validation RMSE: ",round(np.sqrt(metrics.mean_squared_error(Y_validate,Y_validate_predicted)),2))
print("Validation MAE: ",round(metrics.mean_absolute_error(Y_validate,Y_validate_predicted),2))

print('------------------------------------------------')

print("Test R-squared: ",round(metrics.r2_score(Y_test,Y_test_predicted),2))
print("Test Explained Variation: ",round(metrics.explained_variance_score(Y_test,Y_test_predicted),2))
print('Test MAPE:', round(get_mape(Y_test,Y_test_predicted), 2)) 
print('Test Mean Squared Error:', round(metrics.mean_squared_error(Y_test,Y_test_predicted), 2)) 
print("Test RMSE: ",round(np.sqrt(metrics.mean_squared_error(Y_test,Y_test_predicted)),2))
print("Test MAE: ",round(metrics.mean_absolute_error(Y_test,Y_test_predicted),2))

dataframe_predicted = pd.DataFrame(Y_validate.values, columns=['Actual'], index = Y_validate.index)
dataframe_predicted['Predicted'] = Y_validate_predicted

dataframe_predicted[['Actual', 'Predicted']].plot(title = "Figure 2")