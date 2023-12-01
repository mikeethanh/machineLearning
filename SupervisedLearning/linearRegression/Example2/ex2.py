import numpy as np

one = np.ones((x.shape[0],1))
Xbar = np.concatenate((one,x),asix = 1)
j = []

def grad(w):
    N = Xbar.shape[0]
    j.append(5/N*np.linalg.norm(y-Xbar.dot(w),2)**2)
    return j[-1]
def myGD(w_init,grad,eta,loop):
    w = [w_init]
    for it in range(loop):
        w_new = w[-1]-eta*grad(w[-1])
        if cost(w_new)<0.01:
            break
        w.append(w_new)
    return (w,it)

w_init = np.array([[1],[1]])
loop = 10000
eta = 0.0001
interation = np.arrange(0,loop)
(w1,it1) = myGD(w_init , grad,eta,loop)
print()        

          