#set up environment
%matplotlib inline
import numpy as np
import pandas as pd
import pylab as pl
from sklearn.cross_validation import train_test_split as tts
from sklearn.neighbors import KNeighborsClassifier

#reading in the data
df = pd.read_csv("https://s3.amazonaws.com/demo-datasets/wine.csv")

#defining the features we want to analyze
features = ['density', 'sulphates', 'residual_sugar','high_quality']
df2=pd.DataFrame()

#creating new dataframe with just items we want to look at
for item in features:
    df2[item]=df[item]

#splitting training and test data
train, test = tts(df2, train_size=.8)
train_x=train.drop('high_quality', axis=1)
train_y=train['high_quality']
test_x=test.drop('high_quality', axis=1)
test_y=test['high_quality']

len(train_x)
len(test_x)

results=[]
for n in range(1,51):
    knc=KNeighborsClassifier(n_neighbors=n)
    knc.fit(train_x, train_y)
    predictions = knc.predict(test_x)
    accuracy = np.where(predictions==test_y, 1, 0).sum() / float(len(test))
    results.append([n,accuracy])

results=pd.DataFrame(results, columns=['n','accuracy'])

pl.plot(results.n, results.accuracy)
pl.title("Accuracy with Increasing K")
pl.show()
