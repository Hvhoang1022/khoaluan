import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
mushrooms=pd.read_csv('mushrooms_clean.csv')


#feature encoding
df=mushrooms.copy()
target="class"
encode=list(df.loc[:,df.columns!="class"].columns)

for col in encode:
  dummy=pd.get_dummies(df[col],prefix=col)
  df=pd.concat([df,dummy],axis=1)
  del df[col]

target_mapper={"Edible":0, "Poisonous":1}
def target_encode(val):
  return target_mapper[val]

df["class"] = df["class"].apply(target_encode)

#separating X and Y
X = df.iloc[:, 1:]
y = df.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Build RF model

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = knn.predict(X_test)

# Tính chỉ số chính xác
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
#saving the model
import pickle
pickle.dump(knn,open("mushrooms_KNN.pkl","wb"))


