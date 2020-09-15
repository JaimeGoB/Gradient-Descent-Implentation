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
##################################
#Jaime Goicoechea - Caleb Captain
##################################
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

##################################
#5. Develop a Gradient Descent Optimizer Model
##################################

#h_theta(x) = theta_0 + theta(x)T AKA y  = b + mx(T)
def hypothesis_function(X, thetas):
    return (X.dot(thetas))
    
# error = (h_0) - (y_actual)
def compute_error_difference(h_0, Y):
    return (np.subtract(h_0, Y))

#updating gradients from all predictor variables
def update_gradients(error_difference, X):
    return (error_difference.T.dot(X))

#Equation:
#J = (1/2n) * sum( h - y )^ 2
#    PART1        PART2
def loss_function(X, Y, thetas):

    #getting number of observations
    n = float(len(X))
    
    #Hypothesis function (predict the value of y (y_hat) ) 
    #[1 X 428]
    h_0 =hypothesis_function(X, thetas)

    #Sum of loss (sum of squared error) - PART2 Equation 
    #[1 X 428]
    error_difference = compute_error_difference(h_0, Y)
    
    # #Sum all errors from vector 
    # #[1 X 1]
    sum_squared_error = np.sum(np.square(error_difference))
    
    #mean sqaured error - dividing sum by n observations - PART1 Equation 
    #[1 X 1]
    mean_squared_error = sum_squared_error / (n)
        
    #[1 X 1]
    return (mean_squared_error)

#You want to randmly initialize weights to a value close to zero         
def random_initialization_thetas():
   #Creating 1 x p of random thetas(weights)
   return [np.random.normal(0, 0.01, (5,1))]


def Adaptive_Gradient_Descent(X, Y, thetas, learning_rate = 1e-2, iterations = 100, eps = 1e-6):

    #getting number of observations
    n = float(len(X))
    
    #initliazing empty dataframe for sum of gradient sqr
    Sum_Gradient_Squared = 0
    #initliazing empty array to hold loss values
    total_cost = []
    
    for iteration in range(iterations):
        
        for thetas_update in thetas:
            #[428 X 1] - hypothesis function
            h_0 = hypothesis_function(X, thetas_update)
            #[428 X 1] - cost function
            error_difference = compute_error_difference(h_0, Y)
            #[1 X 5] - compute gradient function
            gradients = update_gradients(error_difference, X)
            #Squaring all gradients of weights and adding them together
            #[5 X 1]
            Sum_Gradient_Squared += (gradients.T ** 2)
            #Update thetas
            #[5 x 1]
            thetas_update[:] -=  1/len(X) * gradients.T * ((learning_rate / np.sqrt(Sum_Gradient_Squared + eps)))
            
            #keep track of loss to plot cost vs iterations
            cost = loss_function(X, Y, thetas_update)
            total_cost.append(cost)
        
    return (thetas_update, total_cost)

def Adjusted_R2(X_test, Y_test, final_thetas):
    
    #Hypothesis function (predict the value of y (y_hat) ) 
    #[1 X 428]
    h_0 =hypothesis_function(X_test, final_thetas)    
    
    #Sum of loss (sum of squared error) - PART2 Equation 
    #[1 X 428]
    error_difference = compute_error_difference( Y_test, h_0)
    
    # #Sum all errors from vector 
    # #[1 X 1]
    sum_squared_error = np.sum(np.square(error_difference))    
    
    # #Sum y_actual - y_bar
    # #[1 X 1]
    sum_squared_total = np.sum(np.square(np.subtract(Y_test, (np.mean(Y_test) ) ) ) )
    #keeping r2 between 0-1
    r_squared = abs(1 - (float(sum_squared_error )) / sum_squared_total)
    #adj r2 function
    adj_r_squared = 1 - (1-r_squared)*(len(Y_test)-1)/(len(Y_test)-X_test.shape[1]-1)
    
    return r_squared,adj_r_squared
    
################################################
#Tuning parameters(learning rate, iteration and thetas)
#to achieve the optimum error value. 
################################################

#Creating df to hold optimal values
log_data = pd.DataFrame(columns = {"lr", "iterations", "weights", "mse"})
#arranging in this specic order
log_data = log_data[["weights", "lr", "iterations", "mse"]]

learning_rate_values = [.01, .001, .0001]
number_of_trials = [100, 250]

for n in number_of_trials:
    for lr in learning_rate_values:
        
        #initializing weights to random values.
        random_thetas = random_initialization_thetas()
        fitted_thetas, loss = Adaptive_Gradient_Descent(X_train, Y_train, random_thetas, lr, n)
        
        #plotting cost vs iterations
        plt.figure()
        plt.plot(loss)
        plt.title('AdaGrad')
        plt.xlabel('Training Iterations')
        plt.ylabel('Cost ')
        plt.show()
        
        #adding information about iterations, thetas and mse to log file
        new_row = {"lr": lr, "iterations": n, "weights": fitted_thetas, "mse":loss}
        log_data = log_data.append(new_row, ignore_index=True)
    

################################################
#Creating a log file that indicates parameters used 
#and error/cost for different trials.
#################################################
log_file = open("log.txt", "w")
log_file.write(log_data.to_string())
log_file.close()

##################################################
#6. Apply the model to the test part of the dataset.
##################################################

#Get optimal weights from log data df. 
optimal_thetas = log_data[["weights"]]


#creating empty hashmap to store MSE and respective thetas(theta_0 & theta_1)
test_cost_and_optimal_paramters = {}

#Iterating through all the weights and
#testing ALL OF THEM on the testing set to get lowest cost.
#We will calculate MSE for each set of thetas(theta_0 and theta_1)
#The MSE will be stored in a dictionary with respective weights/thetas.
for i, j in optimal_thetas.iterrows(): 
    
    #extracting weights array from optimal parametes df
    test_weights = optimal_thetas.iloc[i, 0]
    #MSE from these weights on the test set
    test_cost = float(loss_function(X_test, Y_test,test_weights))

    # #convert float to string to store in dictionary and use as key
    test_cost_string = str(test_cost)
    
    # #key will be MSE and value will be the respective theta parameters
    test_cost_and_optimal_paramters.update({test_cost_string: test_weights})


#sort dictionary to get lowest MSE and their respective weights
descending_dict_mse = collections.OrderedDict(sorted(test_cost_and_optimal_paramters.items()))
#mse using the best theta_0 and theta_1 with less errors
final_mse = next(iter(descending_dict_mse))
#Index 0 - theta_0 (intercept)
#Index 1 - theta_1 (slope)
final_thetas = descending_dict_mse[final_mse]
#r2 - relationship between model & response
#ajdr2 - goodness of fit
r2, adjr2 = Adjusted_R2(X_test, Y_test, final_thetas)
final_r2 = str(r2[0])
final_adjr2 = str(adjr2[0])
###############################
#writing a file that contains a full equation to the final model with the line 
#of best fit using the most optomized weights with least MSE.
###############################

predictors = ["intercept", "x1", "x2", "x3", "x4"]

final_model_file = open("final_model_adagrad.txt", "w")

final_model_file.write("Final Model:\n")
for i in range(len(final_thetas)):
    final_model_file.write( str(final_thetas[i]) + predictors[i] + "+")
final_model_file.write("\n\nMSE " + str(final_mse))
final_model_file.write("\n\nR2 " + final_r2)
final_model_file.write("\n\nAdjusted R2 " + str(final_adjr2))

final_model_file.close()












