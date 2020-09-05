import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

#reading file from url
url = 'https://raw.githubusercontent.com/JaimeGoB/Data-Sets/master/SeoulBikeData.csv'

#using encoding flag to solve UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb0 in position 12: invalid start byte
bikes = pd.read_csv(url, encoding= 'unicode_escape')

bikes.isna().sum()

bikes.head()

#
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(bikes['Rented Bike Count'], bins=50)
plt.show()


correlation_matrix = bikes.corr().round(2)
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)
