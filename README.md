# Jaime Goicoechea - Caleb Captain

# Section 0 - How to run the code

### Libraries Pre-Requisites:
pandas, 
numpy, 
matplotlib.pyplot, 
seaborn

### The files cannot be run through command line.

### They have to be run in Spyer IDE from Anaconda.

#### Steps

#### 1) Remove all txt files from the project. 

#### 2) How to run part1.py

  Option 1: Run all the file at once using F5(green play buton).
  
  Option 2: Run the file by parts. Run lines 1-180, then lines 181-209 then 210-262 and finaly lines 263 to end of file
  
  A "log.txt" file be created at the end of running part1.py.
  
  #### The file "log.txt" was just provided as pre-requiste and proof of the "logs" from parameter tunning.
  
  A "final_model_adagrad.txt" file be created at the end of running part1.py.
  
  #### The files "final_model_adagrad.txt" contain the final fitted model and their performance metrics
  
  
  
#### 3) How to run part2.py

  Option 1: Run all the file at once using F5(green play buton).
  
  Option 2: Run the file by parts. Run lines 1-110 and finaly lines 110 to end of file
  
  A "final_model_linear_regression.txt" file be created at the end of running part1.py.
  
  #### The files "final_model_adagrad.txt" contain the final fitted model and their performance metrics
  

# Section 1 - Adaptive Gradient Descent Optimizer - Equations

We can define the following equation as the hypothesis function. Where theta is a matrix containing the relative weights(coefficients in linear regression) and x is matrix of relevant data.

<img src="https://github.com/JaimeGoB/Gradient-Descent-Implentation/blob/master/images/hypothesis_function.png" length = 1000 width="600"/>


The following function is the cost function that calculates total cost of the dataset with current valued parameters(mean squared error).

<img src="https://github.com/JaimeGoB/Gradient-Descent-Implentation/blob/master/images/cost_function.png" length = 300 width="200"/>

The following equations represent the gradient updates with respect to each feature in x. It computes the rate of change with respect to feature theta_i .

<img src="https://github.com/JaimeGoB/Gradient-Descent-Implentation/blob/master/images/gradient_updates.png" length = 400 width="400"/>

The Adaptive Gradient Descent Optimizer also known as AdaGrad when gradients increases the learning rate decreases and when the gradients decreases the learning rate increases. Essentially, the learning rate value changes. The value of the learning rate is constantly changing depending on the gradient updates.
The Adaptive Gradient Descent optimizer update equation is as follow:

<img src="https://github.com/JaimeGoB/Gradient-Descent-Implentation/blob/master/images/adaptive_gradient_descent.png" length = 400 width="400"/>

# Section 2 - ISTANBUL STOCK EXCHANGE Data Set 
xFinal Fitted Model Performance
Data sets includes returns of Istanbul Stock Exchange with seven other international index; SP, DAX, FTSE, NIKKEI, BOVESPA, MSCE_EU, MSCI_EM from Jun 5, 2009 to Feb 22, 2011.
About the dataset:
Target is to predict the price of the Istanbul stock exchange national 100 index (ISE in Lira currency and USD in Dollars). 
From SP 500 return index, Stock market return index of Germany, Stock market return index of UK, Stock market return index of Japan, Stock market return index of Brazil, MSCI European index, MSCI emerging markets index

We performed a thorough and rigorous analysis of the dataset. We started by checking the linearity of the dataset.


We then moved on to apply a heat to choose attributes with correlation higher than 50%.

<p float="left">
  <img src="https://github.com/JaimeGoB/Gradient-Descent-Implentation/blob/master/images/normality_check.png" width=300 height=300   />
  <img src="https://github.com/JaimeGoB/Gradient-Descent-Implentation/blob/master/images/heat_map.png" width=300 height=300  /> 
</p>


# Section 3 - Final Fitted Model Performance

After running our final fitted model with the optimal parameters(thetas/weights). The model performance metrics from our model to the test data set is as follows:
MSE 0.00021469897563200808
R2 0.4634065153987402
Adjusted R2 0.4371029132124039

These values can be found by on the final_model.txt file that is created at the end of the part1.py. The mean squared error is low and R squared and Adjusted R squared are moderately appropriate and very close to the standard model implementation given by the Python Libraries.
Below we can see plots from running the model with different parameters:
For 100 iterations using .01, .001 and .0001 learning rate value.

<p float="left">
  <img src="https://github.com/JaimeGoB/Gradient-Descent-Implentation/blob/master/images/lr.01-100.png" width=200 height=200   />
  <img src="https://github.com/JaimeGoB/Gradient-Descent-Implentation/blob/master/images/lr.001-100.png" width=200 height=200   />
  <img src="https://github.com/JaimeGoB/Gradient-Descent-Implentation/blob/master/images/lr.0001-100.png" width=200 height=200   />
</p>

For 250 iterations using .01, .001 and .0001 learning rate value.
<p float="left">
  <img src="https://github.com/JaimeGoB/Gradient-Descent-Implentation/blob/master/images/lr.01-250.png" width=200 height=200   />
  <img src="https://github.com/JaimeGoB/Gradient-Descent-Implentation/blob/master/images/lr.001-250.png" width=200 height=200   />
  <img src="https://github.com/JaimeGoB/Gradient-Descent-Implentation/blob/master/images/lr.0001-250.png" width=200 height=200   />
</p>

# Section 4 - Final Fitted Model Equations

### Using custom Adaptive Gradient Descent Optimizer:
[0.001023]intercept+[-0.004590]x1 +[0.01299]x2+[0.0001234]x3+[0.009954]x4

### Using standard model implementation given by the Python Libraries:
[0]intercept + [-.2280]x1 + [-.2809] x2 + [1.1369]x3 + [.9066]x4

