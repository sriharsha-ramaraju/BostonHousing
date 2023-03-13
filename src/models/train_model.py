import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import scipy as sp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

test=pd.read_csv(r'E:\Python Projects\BostonHousing\data\processed\test_data.csv')
train=pd.read_csv(r'E:\Python Projects\BostonHousing\data\processed\train_data.csv')

y_train=train['target']
y_test=test['target']

train.drop('target',axis=1,inplace=True)
test.drop('target',axis=1,inplace=True)

########### linear regression
pip_lr=Pipeline([('scaling',StandardScaler()),
                 ('linearReg',LinearRegression(n_jobs=-1))])

pip_lr.fit(train, y_train)
y_pred=pip_lr.predict(test)

lrResults=pd.DataFrame({'metrics':['r2','mse','rmse','mae'],
              'lrResults':[r2_score(y_test,y_pred),
                        mean_squared_error(y_test,y_pred),
                        np.sqrt(mean_squared_error(y_test,y_pred)),
                        mean_absolute_error(y_test,y_pred)]})

############ ridge regression
pip_ridge=Pipeline([('scaling',StandardScaler()),
                 ('linearReg',Ridge())])

pip_ridge.fit(train, y_train)
y_predRidge=pip_ridge.predict(test)

ridgeResults=pd.DataFrame({'metrics':['r2','mse','rmse','mae'],
              'ridgeResults':[r2_score(y_test,y_predRidge),
                        mean_squared_error(y_test,y_predRidge),
                        np.sqrt(mean_squared_error(y_test,y_predRidge)),
                        mean_absolute_error(y_test,y_predRidge)]})

############ lasso
pip_lasso=Pipeline([('scaling',StandardScaler()),
                 ('linearReg',Lasso())])

pip_lasso.fit(train, y_train)
y_predLasso=pip_lasso.predict(test)

lassoResults=pd.DataFrame({'metrics':['r2','mse','rmse','mae'],
              'lassoResults':[r2_score(y_test,y_predLasso),
                        mean_squared_error(y_test,y_predLasso),
                        np.sqrt(mean_squared_error(y_test,y_predLasso)),
                        mean_absolute_error(y_test,y_predLasso)]})

linearModelResults=lrResults.merge(ridgeResults,on='metrics').merge(lassoResults,on='metrics')

###### simple linear regression-ordinary least squares seems to be the best 

#### prediction on single sample
pip_lr.predict(test.iloc[0,:].values.reshape(1,-1))[0]

import pickle
modelPath=open(r'E:\Python Projects\BostonHousing\models\regressionModel.pkl','wb')
pickle.dump(pip_lr,modelPath)

# cheicking if the model predicts well
model=pickle.load(open(r'E:\Python Projects\BostonHousing\models\regressionModel.pkl','rb'))
model.predict(test.iloc[0,:].values.reshape(1,-1))[0]