import os
import pandas as pd
import numpy as np

#reading file from url
url = 'https://raw.githubusercontent.com/JaimeGoB/Data-Sets/master/data_instanbul.csv'
stocks = pd.read_csv(url)

#checking for missing values or NaN
stocks.isna().sum()

stocks.head()

