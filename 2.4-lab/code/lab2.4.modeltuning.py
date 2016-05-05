import numpy as np
import pandas as pd
import pylab as pl
from sklearn.neighbors import KNeighborsClassifier
# allow plots to appear in the notebook
%matplotlib inline
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score


# increase default figure and font sizes for easier viewing
plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['font.size'] = 14
df = pd.read_csv("https://s3.amazonaws.com/demo-datasets/wine.csv")

test_idx = np.random.uniform(0, 1, len(df)) <= 0.8
train = df[test_idx==True]
test = df[test_idx==False]
features = ['density', 'sulphates', 'residual_sugar']

score_dict = {}
k = []
error = []
for i in range(1,51):
    knn = KNeighborsClassifier(n_neighbors = i)
#knn using train set, where first element is train features and second element is train target
    knn.fit(train[features], train.high_quality)
    score = knn.score(test[features], test.high_quality)
    score_dict[i] = score
    k.append(i)
    error.append(score)
    print 'The accuracy_score for k = %d is : %.3f' % (i,score)

print 'The accuracy_score is %d' % max(score_dict, key = score_dict.get)

####logistic regression
#when doing train_test_split, the positions of these 4 are fixed, so first
X2_train, X2_test, y2_train, y2_test = train_test_split(df[features], df['high_quality'], test_size = 0.33)
logreg = linear_model.LogisticRegression()

###this doesnt work because they have different shapes

mdl = logreg.fit(X2_train,y2_train)
logPreds = mdl.predict(X2_test)
accuracy2 = accuracy_score(y2_test, mdl.predict(X2_test))
print accuracy2
