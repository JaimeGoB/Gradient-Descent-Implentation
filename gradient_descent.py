import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 



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
sns.distplot(stocks['ISE(USD)'], bins=30)
plt.show()

#Checking correlation between predictors 
#We will use Emerging Markets Index.
correlation_matrix = stocks.corr().round(2)
#annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)

#In our linear regression model we will use:
#Emerging-Markets-Index to predict ISE price in USD.
stocks.drop(['ISE'], axis=1, inplace = True)
stocks.drop(['date'], axis=1, inplace = True)
stocks.drop(['SP'], axis=1, inplace = True)
stocks.drop(['DAX'], axis=1, inplace = True)
stocks.drop(['FTSE'], axis=1, inplace = True)
stocks.drop(['NIKKEI'], axis=1, inplace = True)
stocks.drop(['BOVESPA'], axis=1, inplace = True)
stocks.drop(['EU'], axis=1, inplace = True)

##################################
#4. Spliting the dataset into training and test parts. 
##################################

#Will be used to hold values of thetas(theta_knot and theta_1)
theta = np.zeros(2)

#Splitting datasets in training and test
train = stocks[:int(len(stocks)*0.85)]
test = stocks[len(train):]

##################################
#5. Develop a Gradient Descent Optimizer Model
##################################

#h_theta(x) = theta_0 + theta_1(x_1) AKA y  = b + mx
def hypothesis_function(t0, t1, x1):
    return (t0 + t1 * x1)

#Equation:
#J = (1/2n) * sum( h - y )^ 2
#    PART1        PART2
def loss_function(train, theta):

    #getting number of observations
    n = float(len(train))
    
    #get m and b(intercept b and slope m)
    theta_0 = theta[0]
    theta_1 = theta[1]
    
    loss = 0
    
    #Will iterate through each point from set provided
    for index, row in stocks.iterrows():
        
        #get x and y from datasets
        x_1 = row['Emerging-Markets-Index']
        #y actual
        y = row['ISE(USD)']
        
        #Hypothesis function (predict the value of y (y_hat) )
        h_0 =hypothesis_function(theta_0, theta_1, x_1)
        
        #Sum of loss (sum of squared error) - PART2 Equation
        loss = loss + ((h_0 - y) ** 2)
        
    #mean sqaured error - dividing sum by n observations - PART1 Equation
    mean_squared_error = loss / n
        
    return mean_squared_error

loss_function(train, theta)



##################################
#6. Apply the model to the test part of the dataset.
##################################






