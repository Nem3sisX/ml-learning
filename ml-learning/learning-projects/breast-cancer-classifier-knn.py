import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, neighbors
import pandas as pd

df=pd.read_csv('Datasets/breast-cancer.data.txt')
df.replace('?',-99999,inplace=True)

df.drop(['id'], 1, inplace=True)

X= np.array(df.drop(['class'],1))
y= np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier(n_jobs=-1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures= np.array([[4,6,1,3,1,8,4,2,2]])
prediction=clf.predict(example_measures)
print(prediction)

if prediction == 2:
    print("Benign Tumour")
elif prediction== 4:
    print("Malignant Tumour")
else:
    print('Invalid Prediction')
