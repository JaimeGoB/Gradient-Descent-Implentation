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
def hypothesis_function(dataset, theta):
  #getting intercept
  t0 = theta[0]
  
  #getting theta1...theta4
  #1 X p
  theta_sub_i_vector = np.array([theta[1:] ])
  
  #p X N
  x_sub_i_vectors = dataset.loc[:, ~dataset.columns.isin(['ISE(USD)'])]
  x_sub_i_vectors = x_sub_i_vectors.to_numpy()
  x_sub_i_vectors = np.transpose(x_sub_i_vectors)

  # 1 x 4 dot 4 x N
  #[1 X 428]
  return (t0 + np.dot(theta_sub_i_vector, x_sub_i_vectors))

theta = [1 , 1, 1, 1, 1]
test_hypothesis = hypothesis_function(train, theta)
test_hypothesis.shape #1 x 428

#Equation:
#J = (1/2n) * sum( h - y )^ 2
#    PART1        PART2
def loss_function(dataset, theta):

    #getting number of observations
    n = float(len(dataset))
    
    #getting y_actual matrix 
    #[1 X 428]
    y = dataset['ISE(USD)']
    y = np.array([y])

    #Hypothesis function (predict the value of y (y_hat) ) 
    #[1 X 428]
    h_0 =hypothesis_function(dataset, theta)

    #Sum of loss (sum of squared error) - PART2 Equation 
    #[1 X 428]
    error_difference = np.subtract(y, h_0)    
    #[1 X 428]
    error_difference = np.square(error_difference)
    # #Sum all errors from vector 
    # #[1 X 1]
    sum_squared_error = np.sum(error_difference)
    #mean sqaured error - dividing sum by n observations - PART1 Equation 
    #[1 X 1]
    mean_squared_error = sum_squared_error / (2 * n)
        
    # #[1 X 1]
    return mean_squared_error

theta = [1 , 1, 1, 1, 1]
test_loss = loss_function(train, theta)
test_loss

#The objective is to minimize loss (error).
#This can be done by calculating the gradient of the loss function.
def compute_gradients_of_loss_function(dataset, theta):

    #initializing the gradients to zero
    #[5 X 1]
    gradients_of_loss_function = np.zeros(5)
    
    #getting number of observations
    n = float(len(dataset))

    #getting y_actual matrix 
    #[1 X 428]
    y = dataset['ISE(USD)']
    y = np.array([y])
    
    #predictor variables 
    #[1 X 428]
    x_1 = dataset['DAX']
    x_1 = np.array([x_1])
    x_2 = dataset['FTSE']
    x_2 = np.array([x_2])
    x_3 = dataset['EU']
    x_3 = np.array([x_3])
    x_4 = dataset['Emerging-Markets-Index']
    x_4 = np.array([x_4])

        
    #Hypothesis function (predict the value of y (y_hat) ) 
    #[1 X 428]
    h_0 =hypothesis_function(dataset, theta)
        
    #[1 X 428]
    cost = np.subtract(y, h_0)
    
    #[1 X 1] 
    gradients_of_loss_function[0] = np.multiply( (-1 / n), np.sum( cost) )
    #[1 X 1] 
    gradients_of_loss_function[1] = np.multiply( (-1 / n), np.sum( np.multiply( x_1, np.transpose(cost) ) ) )
    #[1 X 1] 
    gradients_of_loss_function[2] = np.multiply( (-1 / n), np.sum( np.multiply( x_2, np.transpose(cost) ) ) )
    #[1 X 1] 
    gradients_of_loss_function[3] = np.multiply( (-1 / n), np.sum( np.multiply( x_3, np.transpose(cost) ) ) )
    #[1 X 1] 
    gradients_of_loss_function[4] = np.multiply( (-1 / n), np.sum( np.multiply( x_4, np.transpose(cost) ) ) )
        
    #add episolon to avoid division by zero
    epsilon = 1e-8
    #[5 X 1]
    gradients_of_loss_function = np.divide(gradients_of_loss_function, (n + epsilon) )

    #[5 X 1] all thetas including intercept (intercept + predictors X 1 row)
    return gradients_of_loss_function


theta = [1 , 1, 1, 1, 1]
test_compute = compute_gradients_of_loss_function(train, theta)
test_compute

#Equation of Adaptive gradient descent
#0_t = 0_t-1 - alfa (gradients / sqrt(sum_of_gradients + epsilon) )
def Adaptive_Gradient_Optimizer(dataset, theta, learning_rate = 1e-2, iterations = 3, e = 1e-8):

    #initliazing empty array to hold loss values
    loss = []

    sum_of_squared_gradients = np.zeros(1)

    for t in range(iterations):
        #computing gradients
        #[5]
        gradients = compute_gradients_of_loss_function(dataset, theta)
    
        #square gradients and then add all gradient
        #[1 X 1]
        sum_of_squared_gradients = np.sum( np.square(gradients))
        
        #add episolon to avoid dividing by zero
        gradient_over_ss_gradient = gradients / (np.sqrt(sum_of_squared_gradients + e))
        
        #updating weights in the function
        theta = theta - (learning_rate * gradient_over_ss_gradient)
              
        #keep track of loss
        cost = loss_function(dataset, theta)
        loss.append(cost)

    #[iterations X 1]
    return loss

theta = [1 , 1, 1, 1, 1]
test_loss = Adaptive_Gradient_Optimizer(train, theta)
test_loss

################################################
#Tuning parameters(learning rate, iteration and thetas)
#to achieve the optimum error value. 
################################################
#You want to randmly initialize weights to a value close to zero         
def random_initialization_thetas():

   #Creating 1 x p of random thetas(weights)
   t = np.zeros(5)
   t[0] = np.random.normal(-0.22728302, .1, size=(1,))
   t[1] = np.random.normal(0.15509949, .1, size=(1,))
   t[2] = np.random.normal(-0.35449038, .1, size=(1,))
   t[3] = np.random.normal(0.48699315, .1, size=(1,))
   t[4] = np.random.normal(-0.11458056, .1, size=(1,))

   return t

log_data = pd.DataFrame(columns = {"lr", "iterations", "weights", "mse"})
log_data = log_data[["weights", "lr", "iterations", "mse"]]
learning_rate_values = [.01, .001, .0001]

trials_100 = 100
for j in learning_rate_values:
    
    #initializing weights to random values.
    theta = random_initialization_thetas()

    loss = Adaptive_Gradient_Optimizer(train, theta, j, trials_100)

    # plt.figure()
    # plt.plot(loss)
    # plt.title('AdaGrad')
    # plt.xlabel('Training Iterations')
    # plt.ylabel('Cost ')
    # plt.show()
    
    new_row = {"lr": j, "iterations": trials_100, "weights": theta, "mse":loss}
    log_data = log_data.append(new_row, ignore_index=True)
    
#print(log_data)


trials_250 = 250
for j in learning_rate_values:
    
    #initializing weights to random values.
    theta = random_initialization_thetas()

    loss = Adaptive_Gradient_Optimizer(train, theta, j, trials_250)

    # plt.figure()
    # plt.plot(loss)
    # plt.title('AdaGrad')
    # plt.xlabel('Training Iterations')
    # plt.ylabel('Cost ')
    # plt.show()
    
    new_row = {"lr": j, "iterations": trials_100, "weights": theta, "mse":loss}
    log_data = log_data.append(new_row, ignore_index=True)
    
#print(log_data)
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
    print(theta_hypothesis)
    print(test_cost)
    print("")
    # #convert float to string to store in dictionary and use as key
    test_cost_string = str(test_cost)
    
    # #key will be MSE and value will be the respective theta parameters
    cost_test_and_optimal_paramters.update({test_cost_string: theta_hypothesis})


#sort dictionary to get lowest MSE and their respective weights
descending_dict_mse = collections.OrderedDict(sorted(cost_test_and_optimal_paramters.items()))
#mse using the best theta_0 and theta_1 with less errors
final_mse = next(iter(descending_dict_mse))
#Index 0 - theta_0 (intercept)
#Index 1 - theta_1 (slope)
final_thetas = descending_dict_mse[final_mse]



###############################
#Do not run after here.
###############################


###############################
#change this to add more variables
###############################
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



