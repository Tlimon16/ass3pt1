import  numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
def comp_model(z,t):
    x,y=z
    dx_dt=x*(2-0.4*x-0.3*y)
    dy_dt=y*(1-0.1*y-0.3*x)
    return [dx_dt,dy_dt]

t=np.linspace(0,50,500)
in_condition=[(1.5,3.5),(1,1),(2,7),(4.5,0.5)]
plt.figure(figsize=(6,6))
for i,(x0,y0) in enumerate(in_condition):
   z0=[x0,y0]
   z=odeint(comp_model,z0,t)
   x,y=z.T
   plt.subplot(2,2,i+1)
   plt.plot(t,x,label='X')
   plt.plot(t,y,label='Y')
   plt.xlabel('Time')
   plt.ylabel('Population')
   plt.legend()
plt.show()
