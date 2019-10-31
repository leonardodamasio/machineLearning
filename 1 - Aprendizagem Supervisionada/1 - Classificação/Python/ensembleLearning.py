# Ensemble Learning

## Objetivo: Detecção de fraude

### Autor: Leonardo Vinícius Damasio da Silva | Cientista de Dados
### Data: 31/10/2019

print("""\

                                                                                                
FFFFFFF RRRRRR    AAA   UU   UU DDDDD      DDDDD   EEEEEEE TTTTTTT EEEEEEE  CCCCC  TTTTTTT IIIII  OOOOO  NN   NN 
FF      RR   RR  AAAAA  UU   UU DD  DD     DD  DD  EE        TTT   EE      CC    C   TTT    III  OO   OO NNN  NN 
FFFF    RRRRRR  AA   AA UU   UU DD   DD    DD   DD EEEEE     TTT   EEEEE   CC        TTT    III  OO   OO NN N NN 
FF      RR  RR  AAAAAAA UU   UU DD   DD    DD   DD EE        TTT   EE      CC    C   TTT    III  OO   OO NN  NNN 
FF      RR   RR AA   AA  UUUUU  DDDDDD     DDDDDD  EEEEEEE   TTT   EEEEEEE  CCCCC    TTT   IIIII  OOOO0  NN   NN 
                                                                                                                 
                                                                                            por Leonardo Damasio


*** Ensemble Learning ***

What is an ensemble method?

"Ensemble models in machine learning combine the decisions from multiple models to improve the overall performance." 

                                                                                    Reference: Towards Data Science


*** Models Used ***

Logistic Regression
Naive Bayes Classifier
K-Nearest Neighbors Classifier
Decision Tree Classifier
Support Vector Machine Classifier
Random Forest Classifier
Extreme Gradient Boosted Trees Classifier
Deep Learning Multilayer Perceptron Neural Networks


*** Ensemble Technique ***

Model Score = m
Accuracy = a
Final Score = Weighted Average = ((m1 * a1 + m2 * a2 + ... + mn * an) / n) / (a1 + a2 + ... + an / n)

""")


print("*** Importing tools ***")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils

print("\nSucess\n")


print("*** Creating Results Directory ***")

try: os.mkdir('results')
except OSError: pass

print("\nSucess\n")


print("*** Importing Datasets ***")

n_input = 26
labels = ["v"+str(x) for x in range(n_input)]
random = 0

dataset = pd.read_csv("credit.csv", sep = ",")

x = dataset.iloc[:,0:20]
labels = [i for i in x.columns]
x = x.values
labelencoder = LabelEncoder()
transform = [0,2,3,5,6,8,9,11,13,14,16,18,19]
for i in transform:
    x[:,i] = labelencoder.fit_transform(x[:,i])
x = pd.DataFrame(x)
x.columns = labels

y = dataset.iloc[:,20]
y = y.values
y = labelencoder.fit_transform(y)
y = pd.DataFrame(y)
y.columns = ["class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=random)

new_dataset = pd.read_csv("novocredit.csv", sep = ",")
new_x = new_dataset
new_x = new_x.values
labelencoder = LabelEncoder()
transform = [0,2,3,5,6,8,9,11,13,14,16,18,19]
for i in transform:
    new_x[:,i] = labelencoder.fit_transform(new_x[:,i])
new_x = pd.DataFrame(new_x)
new_x.columns = labels

print("\nSucess\n")


# MODEL_1 Logistic Regression

print("\n*** Running MODEL_1 Logistic Regression ***")

MODEL_1 = LogisticRegression(solver="lbfgs", random_state=random)

MODEL_1.fit(x_train, y_train)
pred_y_test = MODEL_1.predict(x_test)
accuracy_1 = accuracy_score(y_test, pred_y_test)

pred_new_y = MODEL_1.predict(new_x)
new_probability = MODEL_1.predict_proba(new_x)
scores_1 = pd.DataFrame(new_probability[:,1], columns=["MODEL_1_LogisticRegression"])

print("\nModel_1/8: Logistic Regression | Accuracy:", (accuracy_1*100).round(4), "%")


# MODEL_2 Naive Bayes Classifier

print("\n*** Running MODEL_2 Naive Bayes Classifier ***")

MODEL_2 = GaussianNB()

MODEL_2.fit(x_train, y_train)
pred_y_test = MODEL_2.predict(x_test)
accuracy_2 = accuracy_score(y_test, pred_y_test)

pred_new_y = MODEL_2.predict(new_x)
new_probability = MODEL_2.predict_proba(new_x)
scores_2 = pd.DataFrame(new_probability[:,1], columns=["MODEL_2_NaiveBayes"])

print("\nMODEL_2/8: Naive Bayes Classifier | Accuracy:", (accuracy_2*100).round(4), "%")


# MODEL_3 K-Nearest Neighbors Classifier

print("\n*** Running MODEL_3 K-Nearest Neighbors Classifier ***")

MODEL_3 = KNeighborsClassifier(n_neighbors = 3)

MODEL_3.fit(x_train, y_train)
pred_y_test = MODEL_3.predict(x_test)
accuracy_3 = accuracy_score(y_test, pred_y_test)

pred_new_y = MODEL_3.predict(new_x)
new_probability = MODEL_3.predict_proba(new_x)
scores_3 = pd.DataFrame(new_probability[:,1], columns=["MODEL_3_KNN"])

print("\nMODEL_3/8: K-Nearest Neighbors Classifier | Accuracy:", (accuracy_3*100).round(4), "%")


# MODEL_4 Decision Tree Classifier

print("\n*** Running MODEL_4 Decision Tree Classifier ***")

MODEL_4 = DecisionTreeClassifier(random_state=random)

MODEL_4.fit(x_train, y_train)
pred_y_test = MODEL_4.predict(x_test)
accuracy_4 = accuracy_score(y_test, pred_y_test)

pred_new_y = MODEL_4.predict(new_x)
new_probability = MODEL_4.predict_proba(new_x)
scores_4 = pd.DataFrame(new_probability[:,1], columns=["MODEL_4_DecisionTree"])

print("\nMODEL_4/8: Decision Tree Classifier | Accuracy:", (accuracy_4*100).round(4), "%")


# MODEL_5 Support Vector Machine Classifier

print("\n*** Running MODEL_5 Support Vector Machine Classifier ***")

MODEL_5 = SVC(gamma='scale', probability=True, random_state=random)

MODEL_5.fit(x_train, y_train)
pred_y_test = MODEL_5.predict(x_test)
accuracy_5 = accuracy_score(y_test, pred_y_test)

pred_new_y = MODEL_5.predict(new_x)
new_probability = MODEL_5.predict_proba(new_x)
scores_5 = pd.DataFrame(new_probability[:,1], columns=["MODEL_5_SVM"])

print("\nMODEL_5/8: Support Vector Machine Classifier | Accuracy:", (accuracy_5*100).round(4), "%")


# MODEL_6 Random Forest Classifier

print("\n*** Running MODEL_6 Random Forest Classifier ***")

MODEL_6 = RandomForestClassifier(n_estimators = 1000, random_state=random)

MODEL_6.fit(x_train, y_train)
pred_y_test = MODEL_6.predict(x_test)
accuracy_6 = accuracy_score(y_test, pred_y_test)

pred_new_y = MODEL_6.predict(new_x)
new_probability = MODEL_6.predict_proba(new_x)
scores_6 = pd.DataFrame(new_probability[:,1], columns=["MODEL_6_RandomForest"])

print("\nMODEL_6/8: Random Forest Classifier | Accuracy:", (accuracy_6*100).round(4), "%")


# MODEL_7 Extreme Gradient Boosted Trees Classifier

print("\n*** Running MODEL_7 Extreme Gradient Boosted Trees Classifier ***")

x_train = x_train.values
y_train = y_train.values
x_test = x_test.values
y_test = y_test.values
new_x = new_x.values

MODEL_7 = XGBClassifier(base_score=0.5, 
                       booster='gbtree', 
                       colsample_bylevel=1,
                       colsample_bynode=1,
                       colsample_bytree=0.8,
                       gamma=0,
                       learning_rate=0.2, 
                       max_delta_step=0, 
                       max_depth=5,
                       min_child_weight=1, 
                       missing=None, 
                       n_estimators=1000,
                       n_jobs=1,
                       nthread=None, 
                       objective='binary:logistic', 
                       random_state=random,
                       reg_alpha=0, 
                       reg_lambda=1, 
                       scale_pos_weight=1, 
                       seed=None,
                       silent=None, 
                       subsample=0.8, 
                       verbosity=1)

MODEL_7.fit(x_train, y_train)
pred_y_test = MODEL_7.predict(x_test)
accuracy_7 = accuracy_score(y_test, pred_y_test)

pred_new_y = MODEL_7.predict(new_x)
new_probability = MODEL_7.predict_proba(new_x)
scores_7 = pd.DataFrame(new_probability[:,1], columns=["MODEL_7_XGBoost"])

print("\nMODEL_7/8: Extreme Gradient Boosted Trees Classifier | Accuracy:", (accuracy_7*100).round(4), "%")


# MODEL_8 Deep Learning Multilayer Perceptron Neural Networks

print("\n*** Running MODEL_8 Deep Learning Multilayer Perceptron Neural Networks ***")

y_test = np_utils.to_categorical(y_test)     
y_train = np_utils.to_categorical(y_train)

MODEL_8 = Sequential()
MODEL_8.add(Dense(units=156, activation="relu", input_dim=20))
MODEL_8.add(Dropout(0.2))
MODEL_8.add(Dense(units=100, activation="relu"))
MODEL_8.add(Dense(units=80, activation="relu"))
MODEL_8.add(Dense(units=60, activation="relu"))
MODEL_8.add(Dense(units=40, activation="relu"))
MODEL_8.add(Dense(units=20, activation="relu"))
MODEL_8.add(Dense(units=2, activation="softmax"))
MODEL_8.summary()
MODEL_8.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]) 

MODEL_8.fit(x_train, y_train, epochs=12, validation_data=(x_test, y_test))
pred_y_test = MODEL_8.predict(x_test)
y_test_un = [np.argmax(i) for i in y_test] 
pred_y_test_un = [np.argmax(i) for i in pred_y_test]
accuracy_8 = accuracy_score(y_test_un, pred_y_test_un) 

pred_new_y = MODEL_8.predict(new_x)
new_probability = MODEL_8.predict_proba(new_x)
scores_8 = pd.DataFrame(new_probability[:,1], columns=["MODEL_8_NeuralNetworks"])

print("\nMODEL_8/8: Deep Learning Multilayer Perceptron Neural Networks | Accuracy:", (accuracy_8*100).round(4), "%")


print("\n*** Ensemble Weights ***")

print("\nModel_1/8: Logistic Regression | Accuracy:", (accuracy_1*100).round(4), "%")
print("\nMODEL_2/8: Naive Bayes Classifier | Accuracy:", (accuracy_2*100).round(4), "%")
print("\nMODEL_3/8: K-Nearest Neighbors Classifier | Accuracy:", (accuracy_3*100).round(4), "%")
print("\nMODEL_4/8: Decision Tree Classifier | Accuracy:", (accuracy_4*100).round(4), "%")
print("\nMODEL_5/8: Support Vector Machine Classifier | Accuracy:", (accuracy_5*100).round(4), "%")
print("\nMODEL_6/8: Random Forest Classifier | Accuracy:", (accuracy_6*100).round(4), "%")
print("\nMODEL_7/8: Extreme Gradient Boosted Trees Classifier | Accuracy:", (accuracy_7*100).round(4), "%")
print("\nMODEL_8/8: Deep Learning Multilayer Perceptron Neural Networks | Accuracy:", (accuracy_8*100).round(4), "%")


max_avg_score = (accuracy_1 + accuracy_2 + accuracy_3 + accuracy_4 + accuracy_5 + accuracy_6 + accuracy_7 + accuracy_8) / 8
print("\n*** Maximum Possible Average Score ***")
print("\n",max_avg_score,"\n")


resultados = pd.DataFrame(new_dataset.join([scores_1, scores_2, scores_3, scores_4, scores_5, scores_6, scores_7, scores_8]))

resultados["Final_Score"] = ((resultados["MODEL_1_LogisticRegression"]*accuracy_1 + 
	resultados["MODEL_2_NaiveBayes"]*accuracy_2 +
	resultados["MODEL_3_KNN"]*accuracy_3 +
	resultados["MODEL_4_DecisionTree"]*accuracy_4 +
	resultados["MODEL_5_SVM"]*accuracy_5 +
	resultados["MODEL_6_RandomForest"]*accuracy_6 +
	resultados["MODEL_7_XGBoost"]*accuracy_7 +
	resultados["MODEL_8_NeuralNetworks"]*accuracy_8) / 8 ) / max_avg_score

print("\n*** Results ***\n\n", resultados.iloc[:,20:], "\n\n")


print("*** Exporting Results ***")

resultados = resultados.sort_values(by="Final_Score", ascending=False)
resultados.to_csv(r"results\\resultados.csv", index=False, sep="|")
print("\nresultados.csv exported!")


input("\nPress ENTER to exit")