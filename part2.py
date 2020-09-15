import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import random
import collections
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn import metrics


##################################
#3. Data-Preprocessing
##################################

#reading file from url
url = 'https://raw.githubusercontent.com/JaimeGoB/Data-Sets/master/data_instanbul.csv'
stocks = pd.read_csv(url)

#renaming the response variables.
stocks.rename(columns={"ISE.1": "ISE(USD)"}, inplace =True)
stocks.rename(columns={"EM": "Emerging-Markets-Index"}, inplace =True)

#checking for missing values or NaN
stocks.isna().sum()

#Checking if dependent variable has a linear relationship with the attributes.
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(stocks['ISE(USD)'], bins=30).set_title('Linearity Check')
plt.show()

#Checking correlation between predictors 
#We will use Emerging Markets Index.
correlation_matrix = stocks.corr().round(2)
#annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True).set_title('Heat Map')

#Droping ISE because we are interested in ISE(USD)
stocks.drop(['ISE'], axis=1, inplace = True)
stocks.drop(['date'], axis=1, inplace = True)
stocks.drop(['SP'], axis=1, inplace = True)
stocks.drop(['NIKKEI'], axis=1, inplace = True)
stocks.drop(['BOVESPA'], axis=1, inplace = True)


###################
#Normalizing dataset
###################
x_1 = preprocessing.scale(stocks["DAX"])
stocks["DAX"] = x_1
x_2 = preprocessing.scale(stocks["FTSE"])
stocks["FTSE"] = x_2
x_3 = preprocessing.scale(stocks["EU"])
stocks["EU"] = x_3
x_4 = preprocessing.scale(stocks["Emerging-Markets-Index"])
stocks["Emerging-Markets-Index"] = x_4

##################################
#4. Spliting the dataset into training and test parts. 
##################################

#Splitting datasets in training and test
X = stocks[['DAX','FTSE','EU','Emerging-Markets-Index']]
intercept = np.ones(len(X))
X.insert (0, 'intercetp', intercept)

Y = stocks[['ISE(USD)']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=101)


lm = LinearRegression()
lm.fit(X_train, Y_train)
predictions = lm.predict(X_test)
###Coefficients
lm.coef_[0][1]

##################################
#This is the Standard Linear Regresssion Implimentation
##################################

print('MAE:', metrics.mean_absolute_error(Y_test, predictions))
print('MSE:', metrics.mean_squared_error(Y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, predictions)))
print('R^2:', metrics.r2_score(Y_test, predictions))

#This is the Stochastic Gradient Descent Regression Implimentation

#n_samples, n_features = 10, 5
# Always scale the input. The most convenient way is to use a pipeline.
reg = make_pipeline(StandardScaler(), SGDRegressor(max_iter=2000, tol=1e-3))
reg.fit(X_train, Y_train)
#Pipeline(steps=[('standardscaler', StandardScaler()), ('sgdregressor', SGDRegressor())])
pred = reg.predict(X_test)

final_mse = metrics.mean_squared_error(Y_test, predictions)
final_r2 = metrics.r2_score(Y_test, pred)

print('MAE:', metrics.mean_absolute_error(Y_test, predictions))
print('MSE:', final_mse)
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, predictions)))
print('R^2:',final_r2)

###############################
#writing a file that contains a full equation to the final model with the line 
#of best fit using linear regression.
###############################
predictors = ["intercept", "x1", "x2", "x3", "x4"]

final_model_file = open("final_model_linear_regression.txt", "w")

final_model_file.write("Final Model:\n")
for i in range(len(predictors)):
    final_model_file.write( "[" + str(lm.coef_[0][i]) + "]" + predictors[i] + "+")
final_model_file.write("\n\nMSE " + str(final_mse))
final_model_file.write("\n\nR2 " + str(final_r2))

final_model_file.close()