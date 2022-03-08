import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

option = st.sidebar.selectbox(
  'Select one:', ['Heart Attack Dataset','Analysis'])

if option=='Heart Attack Dataset':
  st.header('Heart Attack Dataset')
  st.markdown("""
  A heart attack, also called a myocardial infarction, happens when a part of the heart muscle doesn't get enough blood. Coronary artery disease (CAD) is the main cause of heart attack.
  This project will present the heart attack analytics which provide the possibility rate of the presence of heart disease in the patient.
  * **Data Source:** [https://www.kaggle.com/nareshbhat/health-care-data-set-on-heart-attack-possibility)
  """ )

  img = Image.open("heart-attack.jpg")
  st.image(img, use_column_width = True)

  st.write("- The Heart Analysis Dataset:")
    
  path ='heart.csv'
  df = pd.read_csv('heart.csv')
  df.astype({"sex": object})
  df.astype({"fbs": object})
  df.astype({"restecg": object})
  df.astype({"exang": object})
  row_indexes=df[df['age'] >= 40].index
  df.loc[row_indexes,'elderly']="yes"
  row_indexes=df[df['age'] < 40].index
  df.loc[row_indexes,'elderly']="no"
  df.head(10)
  st.dataframe(df)

  obj = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
  for column in obj:
    st.write(column, ':', df[column].unique(), '\n')

elif option=='Analysis':
  st.write("The analysis of heart attack using different machine learning")
  path ='heart.csv'
  df = pd.read_csv('heart.csv')
  df.astype({"sex": object})
  df.astype({"fbs": object})
  df.astype({"restecg": object})
  df.astype({"exang": object})
  row_indexes=df[df['age'] >= 40].index
  df.loc[row_indexes,'elderly']="yes"
  row_indexes=df[df['age'] < 40].index
  df.loc[row_indexes,'elderly']="no"
  feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
  X = df[feature_cols] 
  y = df.target
  from sklearn.model_selection import train_test_split
  from sklearn.model_selection import train_test_split
  Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
  Xtrain.head()
  ytrain.head()
  models = pd.DataFrame(columns=["Model","Accuracy Score"])

  # GROUP BY ELDERLY AND SEX 
  from sklearn.preprocessing import LabelEncoder
  labelencoder = LabelEncoder()
  df['elderly'] = labelencoder.fit_transform(df['elderly'])
  df.groupby(['elderly','sex']).mean() 
    
  # LOGISTIC REGRESSION
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import accuracy_score
  logreg = LogisticRegression()
  logreg.fit(Xtrain, ytrain)
  ypred = logreg.predict(Xtest)
  score1 = accuracy_score(ytest, ypred)
  st.write("Logistic Regression is equal to", score1)

  # KNN CLASSIFIER 
  from sklearn.neighbors import KNeighborsClassifier
  knn = KNeighborsClassifier()
  knn.fit(Xtrain, ytrain)
  ypred = knn.predict(Xtest)
  score2 = accuracy_score(ytest, ypred)
  st.write("KNeighborsClassifier is equal to", score2)

  # SVM
  from sklearn.svm import SVC
  svc = SVC()
  svc.fit(Xtrain, ytrain)
  ypred = svc.predict(Xtest)
  score3 = accuracy_score(ytest, ypred)
  st.write("SVM is equal to", score3)

  # RANDOM FOREST
  from sklearn.ensemble import RandomForestClassifier
  RandomForest = RandomForestClassifier()
  RandomForest.fit(Xtrain, ytrain)
  ypred = RandomForest.predict(Xtest)
  score4 = accuracy_score(ytest, ypred)
  st.write("Random Forest Classifier is equal to", score4)

  st.write("- From all the scores, the accuracy of all the outputs from the model are between 63 percent to 82 percent.")

  st.header('Conclusion')
  st.write("- Tips to keep your heart healthy.You can prevent heart disease by making these lifestyle choices.")
  img2 = Image.open("caring-heart-healthy-tips.jpg")
  st.image(img2, use_column_width = True)