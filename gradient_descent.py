import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split



##################################
#3. Data-Preprocessing
##################################

#reading file from url
url = 'https://raw.githubusercontent.com/JaimeGoB/Data-Sets/master/data_instanbul.csv'
stocks = pd.read_csv(url)

#renaming the response variables.
stocks.rename(columns={"ISE.1": "ISE(USD)"}, inplace =True)
stocks.rename(columns={"SP": "S&P500"}, inplace =True)


#we will drop the ISE Lira value and only keep ISE USD value.
stocks.drop(['ISE'], axis=1, inplace = True)
stocks.drop(['date'], axis=1, inplace = True)

#checking for missing values or NaN
stocks.isna().sum()

#Checking if dependent variable has a linear relationship with the attributes.
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(stocks['ISE(USD)'], bins=30)
plt.show()

#Checking correlation between predictors 
correlation_matrix = stocks.corr().round(2)
#annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)

##################################
#4. Spliting the dataset into training and test parts. 
##################################

#Splitting up the dataset between predictors and responses.
X = pd.DataFrame(np.c_[stocks['S&P500'], stocks['DAX'], stocks['FTSE'], stocks['NIKKEI'], stocks['BOVESPA'], stocks['EU'], stocks['EM']], columns=['S&P500','DAX', 'FTSE', 'NIKKEI', 'BOVESPA', 'EU', 'EM'])
Y = pd.DataFrame(stocks['ISE(USD)'])

#Splitting datasets in training and test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=9)


##################################
#5. Develop a Gradient Descent Optimizer Model
##################################



##################################
#6. Apply the model to the test part of the dataset.
##################################

