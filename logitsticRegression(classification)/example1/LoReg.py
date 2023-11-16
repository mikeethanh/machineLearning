import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('./fakedatacsv.png')

x1 = data.loc[data['Purchased']==1]
x2 = data.loc[data['Purchased']==0]
x1 = x1.iloc[:,[2,3]].values
x2 = x2.iloc[:,[2,3]].values

# visualization data
plt.scatter(x0[:,0],x0[:,1],c = 'blue')
plt.scatter(x1[:,0],x1[:,1],c = 'red')
plt.xlabel('Tuoi tac')
plt.ylabel('Muc luong')
plt.show()

#  scale data
x = (data.iloc[:,[2,3]]).astype('float64')
y = data.iloc[:,4].values

for i in range(0,len(x)):
    x[i][0] = x[i][0] / 60
    x[i][1] = x[i][1] / 150000

# sigmoid function 
def sigmoid(x):
    return 1/(1+np.exp(-x))

# add columns 1
ones = np.ones(len(x),1)
x = np.concatenate((ones,x),axis=1)
w = np.array([0.,0.1,0.1]).reshape(-1,1)

# lap
numOfIteration = 1000
cost = np.zeros((numOfIteration,1))
learningRate = 0.0001
numloop = []
for i in range(numOfIteration):
    # tinh gia tri du doan
    y_predict = sigmoid(np.dot(x,w))
    # lossFunction
    cost[i] = -np.sum(np.multiply(y,np.log(y_predict))+np.multiply(1-y,np.lo(1-y_predict)))
    # gradientDecent
    w = w - learningRate*np.dot(x.T,y_predict-y) 
    numloop.append(i)
numloop = np.array(numloop)
# sai so khoi dau
print()
