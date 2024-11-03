import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("name.csv")
row_sum=10000
columns_sum=10
sum1=0
selection=[]
first=[0]*columns_sum
zero=[0]*columns_sum
for n in range(0,row_sum):
    ad=0
    max_th=0
    for i in range(0,columns_sum):
        rasbeta=random.betavariate(first[i]+1,zero[i]+1)
        if rasbeta>max_th:
            max_th=rasbeta
            ad=i
    selection.append(ad)
    point=data.values[row_sum,ad]
    if point==1:
        first[ad]=first[ad]+1
    else:
        zero[ad]=zero[ad]+1
    sum1=sum1+point
plt.hist(selection)
plt.show()
