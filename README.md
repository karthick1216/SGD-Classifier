# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
## step 1.Import Necessary Libraries and Load Data
## step 2.Split Dataset into Training and Testing Sets
## step 3.Train the Model Using Stochastic Gradient Descent (SGD)
## step 4.Make Predictions and Evaluate Accuracy
## step 5.Generate Confusion Matrix
## Program:

/*
Program to implement the prediction of iris species using SGD Classifier.

Developed by: KARTHICK S

RegisterNumber: 212224230114
*/

```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
iris=load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())
```
## output:
 ![ex 7 s1](https://github.com/user-attachments/assets/e00ec265-ad67-4be0-abc6-b8c41db16765)
```
print(df.tail())
```
## output:
![ex 7 s2](https://github.com/user-attachments/assets/ddee35b9-516f-45e7-96c2-64392a947b81)

```
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
sgd_clf=SGDClassifier(max_iter=1000,tol=1e-3)
sgd_clf.fit(X_train,y_train)
y_pred=sgd_clf.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")
cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cm)
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
```
## output:
![ex 7 s3](https://github.com/user-attachments/assets/f435c941-ce22-458e-b517-a338d2bb776b)



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
