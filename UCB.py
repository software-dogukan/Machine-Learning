import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("name.csv")

row_sum=10000
columns_sum=10
point=[0]*columns_sum
touch=[0]*columns_sum
sum1=0
sum_point=0
selection=[]
for n in range(0,row_sum):
    ad=0
    max_ucb=0

    for i in range(0,columns_sum):
        if touch[i]>0:
            mean=point[i]/touch[i]
            delta=math.sqrt(3/2*math.log(n)/touch[i])
            ucb=row_sum*10
        else:
            ucb=10000
        if max_ucb<ucb:
            max_ucb=ucb
            ad=i
    selection.append(ad)
    touch[ad]=touch[ad]+1
    point=data.values[row_sum,ad]
    sum_point[ad]=sum_point[ad]+point
    sum1=sum1+point
plt.hist(selection)
plt.show()