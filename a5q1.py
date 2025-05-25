import numpy as np
import matplotlib.pyplot as plt 


# Load the data
data = np.loadtxt("ex2data1.txt", delimiter=",")

X = data[:, 0]  
Y = data[:,1] 

m=len(Y)
X_mean=np.mean(X,axis=0)
X_std=np.std(X,axis=0)
X_norm=(X-X_mean)/X_std

X_b=np.c_[np.ones(m),X_norm]
theta=np.zeros(X_b.shape[1])
alpha=0.01
iterations=1500

def compute_cost(X,Y,theta):
    errors=X @ theta - Y
    return(1/(2*m))*np.dot(errors,errors)

def gradient_descent(X,Y,theta,alpha,iterations):
    for _ in range(iterations):
        gradient=(1/m)*(X.T@(X@ theta-Y))
  
    theta-=alpha*gradient
    return theta
theta=gradient_descent(X_b,Y,theta,alpha,iterations)
print("Learned theta:",theta)