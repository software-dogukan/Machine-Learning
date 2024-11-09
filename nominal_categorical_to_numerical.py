import pandas as pd
from sklearn import preprocessing
data=pd.read_csv("name.csv")
new_data=data.iloc[:,0:1].values#forexample it is text
le=preprocessing.LabelEncoder()
new_data[:,0]=le.fit_transform(data.iloc[:,0])
ohe=preprocessing.OneHotEncoder()
new_data=ohe.fit_transform(new_data).toarray()
