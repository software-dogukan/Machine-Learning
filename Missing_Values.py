import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer as si
data=pd.read_csv("name.csv")
imputer=si(missing_values=np.nan,strategy='mean')
new_data=data.iloc[:,:].values
imputer=imputer.fit(new_data[:,:])
new_data=imputer.transform(new_data[:,:])


