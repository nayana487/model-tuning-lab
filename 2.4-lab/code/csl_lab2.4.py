import numpy as np
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
%matplotlib inline

df = pd.read_csv("https://s3.amazonaws.com/demo-datasets/wine.csv")
df.head()

test_idx = np.random.uniform(0, 1, len(df)) <= 0.8
train = df[test_idx==True]
test = df[test_idx==False]

features = ['density', 'sulphates', 'residual_sugar']
target = ['high_quality']
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(train[features], train[target])
predicts = knn.predict(test[features])
accuracy = np.where(predicts == test['high_quality'], 1, 0).sum()/float(len(test))
print accuracy
knn.score(test[features], test[target])
# Accuracy and score are the same values.


def ideal_k():
    k =[]
    accuracy =[]
    for i in range(1,51):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(train[features], train[target])
        k.append(i)
        accuracy.append(knn.score(test[features], test[target]))
        print 'Using %r neighbors gives you an R^2 score of %.3f.' % (i, knn.score(test[features], test[target]))
        plt.plot(k, accuracy)
ideal_k()
