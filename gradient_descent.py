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
    for index, row in stocks.iterrows():
        
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
        cost = loss_function(data,theta)
        
        #stopping condition
        if len(loss) > 0:
            #check if cost is increasing
            if loss[-1] < cost:
                break 
            
        #add error ONLY IFF IT is decreasing
        loss.append(cost)

    

    return loss

#Use to tune the learning rate
def get_random_learning_rate():
    learning_rate = random.uniform(.0001, .001)
    return learning_rate
                    
#You want to randmly initialize weights to a value close to zero         
def random_initialization_thetaas():
   w = np.random.uniform(low=0.00000005, high=0.000001, size=(2,))
   return w 

theta = random_initialization_thetaas()
lr = get_random_learning_rate()

loss = Adaptive_Gradient_Optimizer(train, theta, 1e-3, 300)

plt.plot(loss)
plt.grid()
plt.title('AMSGrad')
plt.xlabel('Training Iterations')
plt.ylabel('Cost ')


#Best learning rate 0.0005791399329246217 - 0.0009324095198754966

#Best random theta weights and random learning rate
#[8.26281591e-07 6.98455793e-07]
#0.0005914531330325669

#[8.28422997e-07 2.73494230e-07]
#0.0008494101979891405

#[5.90831622e-07 5.35492747e-07]
#0.0009433691160820722

#[7.49175543e-07 6.15714291e-07]
#0.0008999066818002277

#Does not work
#[7.92046831e-07 1.55315672e-07]
#0.00012423721494705947


##################################
#6. Apply the model to the test part of the dataset.
##################################






