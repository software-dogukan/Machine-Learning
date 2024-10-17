import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
lin_reg=LinearRegression()
pol_reg=PolynomialFeatures(degree=4)
data=pd.read_csv("name.csv")
x=data.iloc[:,1:2].values
y=data.iloc[:,-1].values
x_poly=pol_reg.fit_transform(x)
lin_reg.fit(x_poly,y)
print(lin_reg.predict(pol_reg.fit_transform([[6.6]])))