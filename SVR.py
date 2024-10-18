import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler()
svr_reg=SVR(kernel="rbf")
data=pd.read_csv("name.csv")
x=data.iloc[:,1:2].values
y=data.iloc[:,-1].values
sc1=StandardScaler()
sc2=StandardScaler()
x_scaler=sc1.fit_transform(x)
y_scaler=sc2.fit_transform(y)
svr_reg.fit(x_scaler,y_scaler)
print(svr_reg.predict(11))