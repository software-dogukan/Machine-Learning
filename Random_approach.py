import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("name.csv")
import random
row_sum=10000
columns_sum=10
sum1=0
selection=[]
for i in range(0,row_sum):
    ad=random.randrange(columns_sum)
    selection.append(ad)
    point=data.values[row_sum,ad]
    sum1=sum1+point
plt.hist(selection)
plt.show()