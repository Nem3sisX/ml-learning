import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn import preprocessing

desired_width=800

pd.set_option('display.width', desired_width)

np.set_printoptions(linewidth=desired_width)

pd.set_option('display.max_columns',12)

df = pd.read_excel('Datasets/titanic.xls')
df.drop(['body','name','boat'],1 , inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)

def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements=set(column_contents)
            x=0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))
    return df
df = handle_non_numerical_data(df)
print(df.head())


X = np.array(df.drop(['survived'],1).astype(float))
X = preprocessing.scale(X)
Y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_it = np.array(X[i]. astype(float))
    predict_it = predict_it.reshape(-1, len(predict_it))
    prediction_act = clf.predict(predict_it)
    if prediction_act[0] == Y[i]:
        correct+=1

print(correct/len(X))