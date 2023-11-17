import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale

data = pd.read('./fakedatacsv')

X_train = data[['Age','EstimatedSalary']].head(300).values
y_train = data['Purschased'].head(300).values
X_test = data[['Age','EstimatedSalary']].tail(100).values
y_test = data['Purschased'].head(300).tail(100).values
print(X_train)

model = LogisticRegression()
model.fit(X_train,y_train)
y_predict = model.predict(X_train)
print(y_predict)

accuracy_score(y_test,y_predict)

x1 = np.array([X_train[i] for i in range (len(y_train)) if y_predict[i] == 1])
x0 = np.array([X_train[i] for i in range (len(y_train)) if y_predict[i] == 0])

plt.scatter(x0[:,0],x0[:,1],c = 'blue')
plt.scatter(x1[:,0],x1[:,1],c = 'red')
plt.show()