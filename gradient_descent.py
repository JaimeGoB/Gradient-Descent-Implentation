import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import random


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

#Splitting datasets in training and test
train = stocks[:int(len(stocks)*0.85)]
test = stocks[len(train):]

##################################
#5. Develop a Gradient Descent Optimizer Model
##################################

#Will be used to hold values of thetas(theta_knot and theta_1)
theta = np.zeros(2)


#h_theta(x) = theta_0 + theta_1(x_1) AKA y  = b + mx
def hypothesis_function(t0, t1, x1):
    return (t0 + t1 * x1)

#Equation:
#J = (1/2n) * sum( h - y )^ 2
#    PART1        PART2
def loss_function(dataset, theta):

    #getting number of observations
    n = float(len(dataset))
    
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

#The objective is to minimize loss (error).
#This can be done by calculating the gradient of the loss function.
def compute_gradients_of_loss_function(dataset, theta):

    #initializing the gradients to zero
    gradients_of_loss_function = np.zeros(2)
    
    #getting number of observations
    n = float(len(dataset))
    
    #get m and b(intercept b and slope m)
    theta_0 = theta[0]
    theta_1 = theta[1]
        
    #Will iterate through each point from set provided
    for index, row in stocks.iterrows():
        
        #get x and y from datasets
        x_1 = row['Emerging-Markets-Index']
        #y actual
        y = row['ISE(USD)']
        
        #Hypothesis function (predict the value of y (y_hat) )
        h_0 =hypothesis_function(theta_0, theta_1, x_1)
        
        
        gradients_of_loss_function[0] += - (2 / n) * x_1 * ( y - h_0 )
        
        gradients_of_loss_function[1] += - (2 / n) * ( y - h_0 )
    
    
    epsilon = 1e-8
    gradients_of_loss_function = np.divide(gradients_of_loss_function, n + epsilon)

    return gradients_of_loss_function

#Equation of Adaptive gradient descent
#0_t = 0_t-1 - alfa (gradients / sqrt(sum_of_gradients + epsilon) )
def Adaptive_Gradient_Optimizer(data, theta, learning_rate = 1e-2, iterations = 300, e = 1e-8):

    #initliazing empty array to hold loss values
    loss = []
    
    sum_of_squared_gradients = np.zeros(theta.shape[0])

    for t in range(iterations):
        
        #computing gradients
        gradients = compute_gradients_of_loss_function(data, theta)
    
        sum_of_squared_gradients += gradients ** 2
    
        #add episolon to avoid dividing by zero
        gradient_over_ss_gradient = gradients / (np.sqrt(sum_of_squared_gradients + e))
    
        #updating weights in the function
        theta = theta - (learning_rate * gradient_over_ss_gradient)

        #keep track of loss
        loss.append(loss_function(data,theta))

    return loss

#Use to tune the learning rate
def get_random_learning_rate():
    learning_rate = 10 ** random.uniform(-6, 1)
    return learning_rate
                             



theta = np.zeros(2)

lr = get_random_learning_rate()
loss = Adaptive_Gradient_Optimizer(train, theta, lr, 300)


plt.plot(loss)
plt.grid()
plt.title('AMSGrad')
plt.xlabel('Training Iterations')
plt.ylabel('Cost ')


#Good - use as parameter learning rate and iterations
#0.00074307775216506 300
#0.00047196199034994004 500


#(straing line going down 45 angle)
#3.758255895832423e-08
#3.758255895832423e-06
##3.758255895832423e-07
#5.28215963754875e-06  
#2.1597591129217853e-06
#1.8286814084952717e-06
#2.3077811957482714e-05
#0.0003929680460792898
#0.00012880000984335732

#BAD > 0.009



##################################
#6. Apply the model to the test part of the dataset.
##################################






