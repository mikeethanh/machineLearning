import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('./fakedatacsv.png')

x1 = data.loc[data['Purchased']==1]
x2 = data.loc[data['Purchased']==0]
x1 = x1.iloc[:,[2,3]].values
x2 = x2.iloc[:,[2,3]].values

plt.scatter(x0[:,0],x0[:,1],c = 'blue')
plt.scatter(x1[:,0],x1[:,1],c = 'red')
plt.xlabel('Tuoi tac')
plt.ylabel('Muc luong')
plt.show()