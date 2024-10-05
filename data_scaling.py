import pandas as pd
from sklearn.model_selection import tain_test_split
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
data=pd.read_csv("name.csv")
for_x=pd.DataFrame(data=data[:,0:6],index=range(22),columns=["fr","tr","us","age","height","weight"])
for_y=pd.DataFrame(data=data[:,-1],index=range(22),columns=["gender"])
x_train,x_test,y_train,y_test=tain_test_split(for_x,for_y,test_size=0.33,random_state=0)
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)