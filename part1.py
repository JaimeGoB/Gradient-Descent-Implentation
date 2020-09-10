import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import random
import collections
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression


##################################
#3. Data-Preprocessing
##################################

#reading file from url
url = 'https://raw.githubusercontent.com/JaimeGoB/Data-Sets/master/data_instanbul.csv'
stocks = pd.read_csv(url)

#renaming the response variables.
stocks.rename(columns={"ISE.1": "ISE(USD)"}, inplace =True)
stocks.rename(columns={"EM": "Emerging-Markets-Index"}, inplace =True)

#Droping ISE because we are interested in ISE(USD)
stocks.drop(['ISE'], axis=1, inplace = True)
stocks.drop(['date'], axis=1, inplace = True)
stocks.drop(['SP'], axis=1, inplace = True)
stocks.drop(['NIKKEI'], axis=1, inplace = True)
stocks.drop(['BOVESPA'], axis=1, inplace = True)


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
train = stocks[:int(len(stocks)*0.80)]
test = stocks[len(train):]

#splitting the test set to apply final model
x_test = stocks[['Emerging-Markets-Index']]
y_test = stocks[['ISE(USD)']]

##################################
#5. Develop a Gradient Descent Optimizer Model
##################################

#h_theta(x) = theta_0 + theta_1(x_1)T AKA y  = b + mx(T)
def hypothesis_function(t0, theta, dataset):
  #1 X p
  theta_sub_i_vector = theta
  #p X N
  x_sub_i_vectors = dataset.loc[:, ~stocks.columns.isin(['ISE(USD)'])]
  x_sub_i_vectors = x_sub_i_vectors.to_numpy()
  x_sub_i_vectors = np.transpose(x_sub_i_vectors)

  # 1 x 4 dot 4 x N
  #return 1 X N
  return (t0 + np.dot(theta_sub_i_vector, x_sub_i_vectors))

#Equation:
#J = (1/2n) * sum( h - y )^ 2
#    PART1        PART2
def loss_function(intercept, theta, dataset):

    #getting number of observations
    n = float(len(dataset))
    #getting y_actual matrix
    y = dataset['ISE(USD)']
    y = y.to_numpy()
    
    #Hypothesis function (predict the value of y (y_hat) )
    h_0 =hypothesis_function(intercept, theta, dataset)

    #Sum of loss (sum of squared error) - PART2 Equation
    error_difference = np.subtract(y, h_0)    
    error_difference = np.square(error_difference)
    sum_squared_error = np.sum(error_difference)
    
    #mean sqaured error - dividing sum by n observations - PART1 Equation
    mean_squared_error = sum_squared_error / (2 * n)
        
    #return mean_squared_error
    return mean_squared_error

theta = [1 , 1, 1, 1]

intercept = 10

test = loss_function(intercept, theta, train)
test


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
        
        
        gradients_of_loss_function[0] += - (1 / n)  * ( y - h_0 )
        
        gradients_of_loss_function[1] += - (1 / n) * x_1 * ( y - h_0 )
    
    
    epsilon = 1e-8
    gradients_of_loss_function = np.divide(gradients_of_loss_function, n + epsilon)

    return gradients_of_loss_function

        
#Equation of Adaptive gradient descent
#0_t = 0_t-1 - alfa (gradients / sqrt(sum_of_gradients + epsilon) )
def Adaptive_Gradient_Optimizer(data, theta, learning_rate = 1e-2, iterations = 100, e = 1e-8):

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
            
        #add error ONLY IFF IT is decreasing
        loss.append(cost)

    

    return loss

#You want to randmly initialize weights to a value close to zero         
def random_initialization_thetas():
   
   #Creting random theta_0 (intercept) 
   intercept = np.random.normal(66, 1, size=(1,))
   
   #Creating 1 x p of random thetas(weights)
   t = np.zeros(2)
   t[0] = np.random.normal(-.06, .10, size=(1,))
   t[1] = np.random.normal(66, 1, size=(1,))
   t[2] = np.random.normal(-.06, .10, size=(1,))
   t[3] = np.random.normal(66, 1, size=(1,))
   
   return (intercept, t)


################################################
#Tuning parameters(learning rate, iteration and thetas)
#to achieve the optimum error value. 
################################################

log_data = pd.DataFrame(columns = {"lr", "iterations", "weights", "mse"})
log_data = log_data[["weights", "lr", "iterations", "mse"]]

#learning_rate_values = [.01, .001, .0001]
learning_rate_values = [.01]

#Change this
trials_100 = 50



for j in learning_rate_values:
    
    #initializing weights to random values.
    intercept, theta = random_initialization_thetas()

    loss = Adaptive_Gradient_Optimizer(train, theta, j, trials_100)

    plt.figure()
    plt.plot(loss)
    plt.title('AdaGrad')
    plt.xlabel('Training Iterations')
    plt.ylabel('Cost ')
    plt.show()
    
    new_row = {"lr": j, "iterations": trials_100, "weights": theta, "mse":loss}
    log_data = log_data.append(new_row, ignore_index=True)
    

print(log_data)

trials_250 = 250

for j in learning_rate_values:
    
    #initializing weights to random values.
    intercept, theta = random_initialization_thetas()

    loss = Adaptive_Gradient_Optimizer(train, theta, j, trials_250)

    plt.figure()
    plt.plot(loss)
    plt.title('AdaGrad')
    plt.xlabel('Training Iterations')
    plt.ylabel('Cost ')
    plt.show()
    
    new_row = {"lr": j, "iterations": trials_250, "weights": theta, "mse":loss}
    log_data = log_data.append(new_row, ignore_index=True)

print(log_data)
################################################
#Creating a log file that indicates parameters used 
#and error/cost for different trials.
#################################################
log_data.to_csv('log.txt', mode='w', header=True, sep='\t', index=False)

##################################################
#6. Apply the model to the test part of the dataset.
##################################################

#Get optimal parameters from log data df. 
optimal_parameters = log_data.copy(deep=True)
#Dropping MSE, lr and iterations because we only need thetas
optimal_parameters.drop(['iterations'], axis=1, inplace = True)
optimal_parameters.drop(['lr'], axis=1, inplace = True)
optimal_parameters.drop(['mse'], axis=1, inplace = True)


#creating empty hashmap to store MSE and respective thetas(theta_0 & theta_1)
cost_test_and_optimal_paramters = {}

#Iterating through all the optimal weights and
#testing them on the testing set.
#We will calculate MSE for each set of thetas(theta_0 and theta_1)
#The MSE will be stored in a dictionary with respective weights/thetas.
for i, j in optimal_parameters.iterrows(): 
    
    #extracting weights array from optimal parametes df
    theta_hypothesis = optimal_parameters.iloc[i, 0]
    
    test_cost = loss_function(test,theta_hypothesis)
    
    # #convert float to string to store in dictionary and use as key
    test_cost_string = str(test_cost)
    
    # #key will be MSE and value will be the respective theta parameters
    cost_test_and_optimal_paramters.update({test_cost_string: theta_hypothesis})


#sort dictionary to get lowest MSE and their respective weights
descending_dict_mse = collections.OrderedDict(sorted(cost_test_and_optimal_paramters.items()))
#mse using the best theta_0 and theta_1 with less errors
final_mse = next(reversed(descending_dict_mse.keys()))
#Index 0 - theta_0 (intercept)
#Index 1 - theta_1 (slope)
final_thetas = descending_dict_mse[next(reversed(descending_dict_mse.keys()))]

#writing a file that contains a full equation to the final model with the line 
#of best fit using the most optomized weights with least MSE.
with open("final_model.txt", "w") as text_file:
    text_file.write("0_hat = 0_o   +  0_1(x_1) \n")
    text_file.write("0_hat = " + str(final_thetas[0]) + " + " + str(final_thetas[1]) + "(x_1)\n")
    text_file.write("MSE: " + final_mse + "\n")



###############################
#We will plot the Adaptive Gradient Descent Model
#to the financial dataset.
###############################

# plotting the points  
plt.scatter(x_test, y_test)  
# naming the x axis 
plt.xlabel('Emerging Markets Index') 
# naming the y axis 
plt.ylabel('ISE Index(USD)') 
# giving a title to my graph 
plt.title('Final Model Application on Test Set') 
#change params on this one
plt.plot(x_test, final_thetas[0] +  final_thetas[1] * x_test, color = 'red', label ='Final Model Approximation.')
plt.legend(framealpha=1, frameon=True, loc = 'lower right');
# function to show the plot 
plt.show()



