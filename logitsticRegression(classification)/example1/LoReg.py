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
print('sai so khoi dau : '+ str(cost[0]))
# sai so cuoi cung
print('sai so cuoi cung:' +str(cost[999]))
# Trong so cuoi cung
print(w)
print('Loss function')
plt.plot(numloop,cost)
plt.show()

# do chinh xac
a = 0
for i in range (len(y_predict)):
    if y_predict[i] < 0.5 :
        y_predict = 0
    else:
        y_predict = 1
    if y_predict == y[i]:
        a += 1 
print('chinh xac' + str(a/len(y_predict)) + '%')

# hien thi du lieu sau khi phan chia
a1 = []
a0 = []
for i in range (len(y_predict)):
    if y_predict == 1 :
        a1.append(x[i])
    else:
        a0.append(x[i])
z1 = np.array(a1)
z0 = np.array(a0)
plt.scatter(z1[:,0] , z1[:,1],c = 'red')
plt.scatter(z0[:,0],z0[:,1],c = 'blue')
plt.show()

# nhap du lieu du doan 
input = [[20.00,30000.00]]
need_pre = np.array(input)
need_pre[0,0] /= 60
need_pre[0,1] /= 150000
# du doan 
ones = np.ones(len(need_pre),1)
need_pre = np.concatenate((ones,need_pre),axis = 1)
result = sigmoid(np.dot(need_pre,w))
print(result)