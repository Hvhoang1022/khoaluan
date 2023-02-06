import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

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
X=df.drop("class",axis=1)
Y=df["class"]
X_train, X_test, Y_train, y_test = train_test_split(X, Y, test_size=0.3)

#Build RF model

model_svc = LinearSVC()
model = CalibratedClassifierCV(model_svc)
model.fit(X_train,Y_train)
# Đánh giá mô hình trên tập kiểm tra
accuracy = model.score(X_test, y_test)
print("Accuracy: ", accuracy)
#saving the model
import pickle
pickle.dump(model,open("mushrooms_svm.pkl","wb"))