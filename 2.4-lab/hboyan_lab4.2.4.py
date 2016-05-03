import numpy as np
import pandas as pd
import pylab as pl
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("https://s3.amazonaws.com/demo-datasets/wine.csv")

test_idx = np.random.uniform(0, 1, len(df)) <= 0.7
train = df[test_idx==True]
test = df[test_idx==False]
features = ['density', 'sulphates', 'residual_sugar']
Xtrain = train[features]
ytrain = train.high_quality
Xtest = test[features]
ytest = test.high_quality

best_k = 0
best_score = 0
for i in range(1,51):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(Xtrain,ytrain)
    score = knn.score(Xtest,ytest)
    print i,score
    if score>best_score:
        best_k=i
        best_score = score
print 'Best: ',best_k,best_score
