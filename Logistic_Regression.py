import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
s_scaler=StandardScaler()
log_r=LogisticRegression(random_state=0)
data=pd.read_csv("name.csv")
x_train,x_test,y_train,y_test=train_test_split(data[:,1:4].values,data[:,4:].values,test_size=0.33,random_state=0)
X_train=s_scaler.fit_transform(x_train)
X_test=s_scaler.fit_transform(x_test)
log_r.fit(x_train,y_train)
y_pred=log_r.predict(X_test)
print(y_pred)
cm=confusion_matrix(y_test,y_pred)
print(cm)