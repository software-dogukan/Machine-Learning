import pandas as pd
import numpy as np
from sklearn.model_selection import tain_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor2=LinearRegression()
le=preprocessing.LabelEncoder()
ohe=preprocessing.OneHotEncoder()
data=pd.read_csv("name.csv")
countries=data.iloc[:,0:1].values
countries[:0]=le.fit_transform(data.iloc[:,0])
countries=ohe.fit_transform(countries).toarray()
gender=data.iloc[:,-1]
gender[:-1]=le.fit_transform(data.iloc[:,-1])
gender=ohe.fit_transform(gender).toarray()
finally1=pd.DataFrame(data=countries,index=range(22),columns=["Tr","Fr","Us"])
finally2=pd.DataFrame(data=data,index=range(22),columns=["age","height","weight"])
finally3=pd.DataFrame(data=gender,index=range(22),columns=["Male","Female"])
combination1=pd.concat([finally1,finally2],axis=1)
combination2=pd.concat([combination1,finally3[:,0:1]],axis=1)
for_x=pd.DataFrame(data=combination1,index=range(22),columns=["Tr","Fr","Us","age","height","weight"])
for_y=pd.DataFrame(data=data[:,-1],index=range(22),columns=["Gender"])
x_train,x_test,y_train,y_test=tain_test_split(for_x,for_y,test_size=0.33,random_state=0)
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
#example 2
weight=combination2.iloc[:,3:4].values
left=combination2.iloc[:,:3]
right=combination2.iloc[:,4:]
data=pd.concat([left,right],axis=1)
x_train,x_test,y_train,y_test=tain_test_split(data,weight,test_size=0.33,random_state=0)
regressor2.fit(x_train,y_train)
y_pred=regressor2.predict(x_test)
#hangi p value değeri yüksek onu bulmamızı ve o değerleri o sütunu çıkarmamıza yaradı
import stadtsmodels.api as sm
X=np.append(arr=np.ones((22,1)).astype(int),values=data,axis=1)
X_ls=data.iloc[:,[0,1,2,3,4,5,]].values
X_ls=np.array(X_ls,dtype=float)
model=sm.OLS(weight,X_ls).fit()
print(model.summary())
