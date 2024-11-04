import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
reviews=pd.read_csv("Reviews.csv")
import re
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
compilation=[]
ps=PorterStemmer()
for i in range(1000):
    review=re.sub("[a-zA-Z]"," ",reviews["Review"[i]])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
    review=" ".join(review)
    compilation.append(review)
#veri ön işlme bitti
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2000)
x=cv.fit_transform(compilation).toarray()
y=reviews.iloc[:,1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(x_train,y_train)
y_pred=gnb.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
