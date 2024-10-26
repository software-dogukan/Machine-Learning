import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("name.csv")
x=data.iloc[:,2:4].values
from sklearn.cluster import KMeans
finally1=[]
for i in range(1,10):
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=12)
    kmeans.fit(x)
    finally1.append(kmeans.inertia_)
plt.plot(range[1,10],finally1)

kmeans=KMeans(n_clusters=3,init="k-means++")