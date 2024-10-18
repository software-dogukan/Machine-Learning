import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor(random_state=0)
data=pd.read_csv("name.csv")
x=data.iloc[:,1:2].values
y=data.iloc[:,-1].values
sc1=StandardScaler()
sc2=StandardScaler()
x_scaler=sc1.fit_transform(x)
y_scaler=sc2.fit_transform(y)
r_dt.fit(x_scaler,y_scaler)
print(r_dt.predict(11))