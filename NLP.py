import numpy as np
import pandas as pd
reviews=pd.read_csv("Reviews.csv")
import re
review=re.sub("[a-zA-Z]"," ",reviews["Review"[0]])
review=review.lower()
review=review.split()
import nltk
nltk.download("stopwords")
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()