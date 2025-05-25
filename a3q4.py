import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp,odeint
L=2
g=32.17
def com_model(theta,t):
    dtheta=theta[1]
    d2theta=-(g/L)*np.sin(theta[0])
    return [dtheta,d2theta]
theta0=(np.pi/6,0)
t=np.arange(0,2.1,0.1)
theta=odeint(com_model,theta0,t)
plt.plot(t,theta[:,0],label='Angle')
plt.title( 'swinging pendulum')
plt.xlabel('Time')
plt.ylabel('Theta')
plt.legend()
plt.grid()
print(theta)
plt.show()