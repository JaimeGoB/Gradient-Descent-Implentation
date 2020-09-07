import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import random
import collections

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

#splitting the test set to apply final model
x_test = stocks[['Emerging-Markets-Index']]
y_test = stocks[['ISE(USD)']]

##################################
#5. Develop a Gradient Descent Optimizer Model
##################################
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
    for index, row in dataset.iterrows():
        
        #get x and y from datasets
        x_1 = row['Emerging-Markets-Index']
        #y actual
        y = row['ISE(USD)']
        
        #Hypothesis function (predict the value of y (y_hat) )
        h_0 =hypothesis_function(theta_0, theta_1, x_1)
        
        #Sum of loss (sum of squared error) - PART2 Equation
        loss = loss + ((h_0 - y) ** 2)
        
    #mean sqaured error - dividing sum by n observations - PART1 Equation
    mean_squared_error = loss / (2 * n)
        
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
    for index, row in dataset.iterrows():
        
        #get x and y from datasets
        x_1 = row['Emerging-Markets-Index']
        #y actual
        y = row['ISE(USD)']
        
        #Hypothesis function (predict the value of y (y_hat) )
        h_0 =hypothesis_function(theta_0, theta_1, x_1)
        
        
        gradients_of_loss_function[0] += - (2 / n)  * ( y - h_0 )
        
        gradients_of_loss_function[1] += - (2 / n) * x_1 * ( y - h_0 )
    
    
    epsilon = 1e-8
    gradients_of_loss_function = np.divide(gradients_of_loss_function, n + epsilon)

    return gradients_of_loss_function

        
#Equation of Adaptive gradient descent
#0_t = 0_t-1 - alfa (gradients / sqrt(sum_of_gradients + epsilon) )
def Adaptive_Gradient_Optimizer(data, theta, learning_rate = 1e-2, iterations = 1, e = 1e-8):

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
        cost = loss_function(data,theta)
        
        #stopping condition
        if len(loss) > 0:
            #check if cost is increasing
            if loss[-1] < cost:
                break 
            
        #add error ONLY IFF IT is decreasing
        loss.append(cost)

    

    return loss
                  
#You want to randmly initialize weights to a value close to zero         
def random_initialization_thetaas():
   w = np.random.uniform(low=0.00000005, high=0.000001, size=(2,))
   return w 

################################################
#Tuning parameters(learning rate, iteration and thetas)
#to achieve the optimum error value. 
################################################

log_data = pd.DataFrame(columns = {"lr", "iterations", "weights", "mse"})
log_data = log_data[["weights", "lr", "iterations", "mse"]]

learning_rate_values = [.01, .001, .0001, .00001]

trials_100 = 100

for j in learning_rate_values:
    
    #initializing weights to random values.
    theta = random_initialization_thetaas()

    loss = Adaptive_Gradient_Optimizer(train, theta, j, trials_100)

    # plt.plot(loss)
    # plt.grid()
    # plt.title('AdaGrad')
    # plt.xlabel('Training Iterations')
    # plt.ylabel('Cost ')
    new_row = {"lr": j, "iterations": trials_100, "weights": theta, "mse":loss}
    log_data = log_data.append(new_row, ignore_index=True)

trials_250 = 250

for j in learning_rate_values:
    
    #initializing weights to random values.
    theta = random_initialization_thetaas()

    loss = Adaptive_Gradient_Optimizer(train, theta, j, trials_250)

    # plt.plot(loss)
    # plt.grid()
    # plt.title('AdaGrad')
    # plt.xlabel('Training Iterations')
    # plt.ylabel('Cost ')
    
    new_row = {"lr": j, "iterations": trials_250, "weights": theta, "mse":loss}
    log_data = log_data.append(new_row, ignore_index=True)
  
################################################
#Creating a log file that indicates parameters used 
#and error/cost for different trials.
#################################################
log_data.to_csv('log.txt', mode='w', header=True, sep='\t', index=False)

##################################################
#6. Apply the model to the test part of the dataset.
##################################################

#Get optimal parameters from log data df. 
optimal_parameters = log_data
#Dropping MSE and iterations because we only learning rate and weights
optimal_parameters.drop(['mse'], axis=1, inplace = True)
optimal_parameters.drop(['iterations'], axis=1, inplace = True)
optimal_parameters.drop(['lr'], axis=1, inplace = True)

#creating empty hashmap to store MSE and respective thetas(theta_0 & theta_1)
cost_test_and_optimal_paramters = {}

#Iterating through all the optimal weights and
#testing them on the testing set.
#We will calculate MSE for each set of thetas(theta_0 and theta_1)
#The MSE will be stored in a dictionary with respective weights/thetas.
for i, j in optimal_parameters.iterrows(): 
    
    #extracting weights array from optimal parametes df
    theta_hypothesis = optimal_parameters.iloc[i, 0]
    
    #computing mean squared error using optimal parameters and testing dataset
    test_cost = loss_function(test,theta_hypothesis)

    #updating MSE because we divided by 2 in loss function
    test_cost = test_cost * 2
    
    #convert float to string to store in dictionary and use as key
    test_cost_string = str(test_cost)
    
    #key will be MSE and value will be the respective theta parameters
    cost_test_and_optimal_paramters.update({test_cost_string: theta_hypothesis})



#sort dictionary to get lowest MSE and their respective weights
cost_test_and_optimal_paramters = collections.OrderedDict(sorted(cost_test_and_optimal_paramters.items()))

#mse using theta_0 and theta_1
final_mse = list(cost_test_and_optimal_paramters.keys())[0] 
  
#Index 0 - theta_0 (intercept)
#Index 1 - theta_1 (slope)
final_thetas = list(cost_test_and_optimal_paramters.values())[0]


#Creating strings to output in final model text file
final_model_equ = "0_hat = 0_o   +  0_1(x_1)"
final_model = "0_hat = " + str(final_thetas[0]) + " + " + str(final_thetas[1]) + "(x_1)"
final_model_mse = "MSE: " + str(final_thetas[1])

#writing a file that contains a full equation to the final model with the line 
#of best fit using the most optomized weights with least MSE.
with open("final_model.txt", "w") as text_file:
    text_file.write(final_model_equ + "\n")
    text_file.write(final_model + "\n")
    text_file.write(final_model_mse + "\n")



