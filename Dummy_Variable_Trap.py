import pandas as pd
from sklearn.model_selection import tain_test_split
from sklearn import preprocessing
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
for_x=pd.DataFrame(data=data[:,0:6],index=range(22),columns=["fr","tr","us","age","height","weight"])
for_y=pd.DataFrame(data=data[:,-1],index=range(22),columns=["gender"])
x_train,x_test,y_train,y_test=tain_test_split(for_x,for_y,test_size=0.33,random_state=0)