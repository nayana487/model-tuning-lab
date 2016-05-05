import numpy as np
import pandas as pd
import pylab as pl
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

df = pd.read_csv("https://s3.amazonaws.com/demo-datasets/wine.csv")

test_idx = np.random.uniform(0, 1, len(df)) <= 0.8
train = df[test_idx==True]
test = df[test_idx==False]

features = ['density', 'sulphates', 'residual_sugar']
y_train = train['high_quality']
X_train = train[features]
y_test = test['high_quality']
X_test = test[features]

# K nearest neighbors classification regression

for i in range(1,51):
    knn = KNeighborsClassifier(n_neighbors=i)
    fit = knn.fit(X_train, y_train)
    test_results = fit.predict(X_test)
    accuracy = np.where(test_results==y_test, 1, 0).sum()/float(len(test_results))
    print "With a K value of",
    print i,
    print "the accuracy is: ",
    print (accuracy*100)
    print '%'


# Logistic Regression
X2_train, X2_test, y2_train, y2_test = train_test_split(df[features], df['high_quality'], test_size=0.33)

logreg = LogisticRegression()
mdl = logreg.fit(X2_train, y2_train)

logPreds = mdl.predict(X2_test)
confusion = np.array(confusion_matrix(y2_test, logPreds))
accuracy2 = accuracy_score(y2_test, logPreds)
