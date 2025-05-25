import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint

def func(y,t):
    dydt=1+(t-y)**2
    return dydt
y0=1
t=np.linspace(2,3,100)
ext=t+1/(1-t)

solve1=odeint(func,y0,t)
solve2=solve_ivp(func,(2,3),[1],method='RK45',t_eval=t)
y1=solve2.y[0]
t1=solve2.t
plt.figure()
plt.subplot(2,2,1)
plt.title(' solution')
plt.plot(t,solve1,'b--')
plt.plot(t,ext,color='r')
plt.plot(t1,y1,'g--')
plt.subplot(2,2,2)
plt.title('error for odeint')
ode_error=np.abs(solve1.flatten()-ext)
plt.plot(t,ode_error,'g--')
plt.subplot(2,2,3)
plt.title('error for solveivp')
ivp_error=np.abs(y1.flatten()-ext)
plt.plot(t1,ivp_error,'y--')



plt.show()
