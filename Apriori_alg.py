import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("name.csv")
t=[]
for i in range(0,7501):
    t.append([str(data.values[i,j])for j in range(0,20)])
from apyori import apriori
kurallar=apriori(t,min_support=0.01,min_confidence=0.2,min_lift=3,min_length=2)
print(list(kurallar))


