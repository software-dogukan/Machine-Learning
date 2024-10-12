import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import tain_test_split
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
data=pd.read_csv("name.csv")
for_x=data["months"]
for_y=data["sales"]
x_train,x_test,y_train,y_test=tain_test_split(for_x,for_y,test_size=0.33,random_state=0)
#veri ön işleme bitti sırada model oluşturma
lr.fit(x_train,y_train)
prediction=lr.predict(x_test)
x_train=x_train.sort_index()
y_train=y_train.sort_index()
plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))




