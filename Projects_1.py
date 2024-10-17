import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data=pd.read_csv("name.csv")
data.apply(preprocessing.LabelEncoder().fit_transform())
col=data.iloc[:,:1]#forexample there are more than two values
col=pd.DataFrame(data=col,index=range(14),columns=["o","r","s"])#forexample there are 14 lines and 3 columns
alldata=pd.concat(col,data.iloc[:,1:],axis=1)
x_train,x_test,y_train,y_test=train_test_split(alldata.iloc[:,:-1],alldata[:,-1:],test_size=0.33,random_state=0)
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
import statsmodels.api as sm
X=np.append(arr=np.ones((14,1)).astype(int),values=alldata.iloc[:,:-1],axis=1)
X_lst=alldata.iloc[:,[0,1,2,3,4,5].values]
X_lst=np.array(X_lst,dtype=float)
model=sm.OLS(alldata.iloc[:-1:],X_lst).fit()
print(model.sumarry())
#forexample there are more then zero values for p values
alldata=alldata.iloc[:,1:]
x_train=x_train.iloc[:,1:]
x_test=x_test.iloc[:,1:]
#and we try again for true prediction
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)


