#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import cufflinks as cf
cf.set_config_file(offline=True)


# In[4]:


from sklearn.model_selection import train_test_split


# In[3]:


cf.datagen.lines().iplot(kind='scatter',xTitle='Dates',yTitle='Returns',title='Cufflinks - Line Chart')


# In[316]:


data = pd.read_excel(r'C:\Users\caleb\Desktop\data_akbilgic.xlsx', skiprows=1)


# In[73]:





# In[6]:


data


# In[70]:


data.info()


# In[72]:


sns.pairplot(data, diag_kind='hist')


# In[86]:


sns.heatmap(data.corr())


# In[76]:


data.columns


# In[79]:


data.corr()


# In[358]:


X = data[['DAX','FTSE','EU','EM']]


# In[8]:


Y = data[['ISE.1']]


# In[330]:


X_train.mean()


# In[331]:


X_train.std()


# In[88]:


data.corr().iplot(kind='heatmap',colorscale="Blues",title="Feature Correlation Matrix")


# In[93]:


X.scatter_matrix()


# In[471]:


def initialize_adagrad_thetas(feature_dim):
    s_w = np.zeros((feature_dim-1,1))
    s_b = np.zeros(1)
    return (s_w, s_b)
#need to change results into something else


# In[473]:


w = np.random.normal(0, 0.01, (5,1))


# In[332]:


len(X)


# In[469]:


b = np.zeros(1)


# In[419]:


params


# In[470]:


params = [w]


# In[1771]:


def gradient(X, Y, )
    for iter = 1:num_iters
        temp = X * theta;
        error = temp - y;
        newX = error' * X;
        theta = theta - ((alpha/m) * newX');


# In[426]:


X_train


# In[220]:


Y_train


# In[346]:


ones = [1,1,1,1]


# In[366]:


np.ones(428)


# In[376]:


new = np.ones(len(Y_train))
new = X
new['intercept'] = 1
new
X = new


# In[474]:


def adagrad(X, Y, states, params, num_iters):
    eps = 1e-6 

    for iter in range(num_iters):
        
        for Theta_new, GradSqr in zip(params, states):
            temp = X.dot(Theta_new);
            error = np.subtract(temp, Y);
            newX = error.T.dot(X)
            GradSqr[:] += newX.T ** 2
            Theta_new[:] -=  1/len(X) * newX.T * ((.01 / np.sqrt(GradSqr + eps)))
            print("Iteration number is: {iter}".format(iter = iter))
    return(Theta_new)
    
        print("Theta is")
        print(Theta_new) 
        


# In[377]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=101)


# In[475]:


adagrad(X_train, Y_train, initialize_adagrad_thetas(6), params, 300)


# In[115]:


[[ 0.00155141]
 [-0.00093958]
 [-0.01263281]
 [ 0.00336985]
 [ 0.00127929]]


# In[51]:


x = np.random.normal(0, 0, size=(1,))


# In[105]:


g = [X, Y]


# In[113]:


data[['ISE.1']].iplot(kind='scatter', mode = 'markers')
data[['DAX' , 'FTSE', 'EU','EM', 'ISE.1']].iplot(kind='surface', mode = 'markers')


# In[15]:


w = np.zeros(2)
w[0] = 0
w[1] = np.random.normal(0.001, 1, size=1)
print(w)
print(w - 1)
w.shape[0]


# In[37]:


X_train.describe()


# In[36]:


X_train.std()


# In[89]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import random
import collections
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
#from sortedcontainers import SortedDict

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
#correlation_matrix = stocks.corr().round(2)
#annot = True to print the values inside the square
#sns.heatmap(data=correlation_matrix, annot=True).set_title('Heat Map')

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
x_test = stocks['Emerging-Markets-Index']
y_test = stocks['ISE(USD)']

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
        loss = loss + ((y - h_0) ** 2)
        
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
        h_0 = hypothesis_function(theta_0, theta_1, x_1)
        
        
        gradients_of_loss_function[0] += - (1 / n)  * ( y - h_0 )
        #print(gradients_of_loss_function[0])
        gradients_of_loss_function[1] += - (1 / n) * x_1 * ( y - h_0 )
    
    
    epsilon = 1e-8
    #gradients_of_loss_function = np.divide(gradients_of_loss_function, n + epsilon)

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

#w = np.random.uniform(low=0.00000005, high=0.000001, size=(2,))                  
#You want to randmly initialize weights to a value close to zero         
def random_initialization_thetaas():
   w = np.zeros(2)
   #w = np.random.normal(0.001, 1, size=(2,))
   w[0] = np.zeros(1)
   w[1] = np.random.normal(0.001, 1, size= 1)
   #w[0] = np.random.uniform(low=0.0000009, high=.0001, size=(1,))
   #w[1] = np.random.uniform(low= - 2.5, high=2.5, size=(1,))
   return w 

def error_difference(dataset, theta):
    
    #get m and b(intercept b and slope m)
    theta_0 = theta[0]
    theta_1 = theta[1]
    
    error_diff = []
    
    #Will iterate through each point from set provided
    for index, row in dataset.iterrows():
        
        #get x and y from datasets
        x_1 = row['Emerging-Markets-Index']
        #y actual
        y = row['ISE(USD)']
        
        #Hypothesis function (predict the value of y (y_hat) )
        h_0 = hypothesis_function(theta_0, theta_1, x_1)
        
        error_diff.append(h_0 - y)
        
    return error_diff

################################################
#Tuning parameters(learning rate, iteration and thetas)
#to achieve the optimum error value. 
################################################


###############################
#run this
###############################
log_data = pd.DataFrame(columns = {"lr", "iterations", "weights", "mse"})
log_data = log_data[["weights", "lr", "iterations", "mse"]]

learning_rate_values = [.01, .001, .0001, .00001]

trials_100 = 100

for j in learning_rate_values:
    
    #initializing weights to random values.
    theta = random_initialization_thetaas()

    loss = Adaptive_Gradient_Optimizer(train, theta, j, trials_100)

    plt.plot(loss)
    plt.grid()
    plt.title('AdaGrad')
    plt.xlabel('Training Iterations')
    plt.ylabel('Cost ')
    new_row = {"lr": j, "iterations": trials_100, "weights": theta, "mse":loss}
    log_data = log_data.append(new_row, ignore_index=True)
    
print(log_data)

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
    
    ###############################
    #CHANGED HERE 
    ###############################
    #test_cost = error_difference(test,theta_hypothesis)
    test_cost = loss_function(test,theta_hypothesis)
    
    # #convert float to string to store in dictionary and use as key
    test_cost_string = str(test_cost)
    
    # #key will be MSE and value will be the respective theta parameters
    cost_test_and_optimal_paramters.update({test_cost_string: theta_hypothesis})
    ###############################
    #TO HERE
    ###############################

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
#This is true intercept from library lm
#plt.plot(x_test, -7.754878291608648e-06+  1.6143252080243709 * x_test, color = 'blue', label = 'Line of Best Fit.')

#change params on this one
plt.plot(x_test, final_thetas[0] +  final_thetas[1] * x_test, color = 'red', label ='Final Model Approximation.')
plt.legend(framealpha=1, frameon=True, loc = 'lower right');
# function to show the plot 
plt.show()







# In[90]:



m , b = np.polyfit(x_test, y_test, 1)
print(m)
print(b)

plt.scatter(x_test, y_test)  
# naming the x axis 
plt.xlabel('Emerging Markets Index') 
# naming the y axis 
plt.ylabel('ISE Index(USD)') 
# giving a title to my graph 
plt.title('Final Model Application on Test Set') 
#This is true intercept from library lm
#plt.plot(x_test, -7.754878291608648e-06+  1.6143252080243709 * x_test, color = 'blue', label = 'Line of Best Fit.')

plt.plot(x_test, m * x_test + b)
plt.plot(x_test, final_thetas[0] +  final_thetas[1] * x_test, color = 'red', label ='Final Model Approximation.')
plt.legend(framealpha=1, frameon=True, loc = 'lower right');
# function to show the plot 
plt.show()


# In[ ]:




