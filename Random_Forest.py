import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
rf_dt=RandomForestRegressor(n_estimators=10,random_state=0)
data=pd.read_csv("name.csv")
x=data.iloc[:,1:2].values
y=data.iloc[:,-1].values
sc1=StandardScaler()
sc2=StandardScaler()
x_scaler=sc1.fit_transform(x)
y_scaler=sc2.fit_transform(y)
rf_dt.fit(x_scaler,y_scaler)
print(rf_dt.predict(11))