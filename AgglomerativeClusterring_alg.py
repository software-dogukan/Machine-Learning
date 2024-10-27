import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("name.csv")
x=data.iloc[:,2:4].values
from sklearn.cluster import AgglomerativeClustering
ac=AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="ward")
y_predict=ac.fit_predict(x)
plt.scatter(x[y_predict==0,1],x[y_predict==0,1],s=100,c="red")
plt.scatter(x[y_predict==1,1],x[y_predict==1,1],s=100,c="red")
plt.scatter(x[y_predict==2,1],x[y_predict==2,1],s=100,c="red")
plt.show()