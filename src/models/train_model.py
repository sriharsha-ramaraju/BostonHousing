import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import scipy as sp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.pipeline import Pipeline

test=pd.read_csv(r'E:\Python Projects\BostonHousing\data\processed\test_data.csv')
train=pd.read_csv(r'E:\Python Projects\BostonHousing\data\processed\train_data.csv')

y_train=train['target']
y_test=test['target']

x_train=train.drop('target',axis=1,inplace=True)
x_test=test.drop('target',axis=1,inplace=True)

