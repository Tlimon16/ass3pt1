import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint

def func(y,t):
    dydt=t*np.exp(3*t)-2*y
    return dydt
y0=0
t=np.linspace(0,1,100)
ext=(1/5)*t*np.exp(3*t)-(1/25)*np.exp(3*t)+(1/25)*np.exp(-2*t)

solve1=odeint(func,y0,t)
solve2=solve_ivp(func,(0,1),[0],method='RK45',t_eval=t)
y1=solve2.y[0]
t1=solve2.t
plt.figure()
plt.subplot(2,2,1)
plt.plot(t,solve1,'b--')
plt.plot(t,ext,color='r')
plt.plot(t1,y1,'g--')
plt.subplot(2,2,2)
ode_error=np.abs(solve1.flatten()-ext)
plt.plot(t,ode_error,'g--')
plt.subplot(2,2,3)
ivp_error=np.abs(y1.flatten()-ext)
plt.plot(t1,ivp_error,'y--')



plt.show()
