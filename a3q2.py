import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def lotka_volterra(t, z):
    x, y = z
    dxdt = -0.1 * x + 0.02 * x * y
    dydt = 0.2 * y - 0.025 * x * y
    return [dxdt, dydt]

t_vals = np.linspace(0, 50, 500)
z0 = [6, 6]

sol = solve_ivp(lotka_volterra, [t_vals[0], t_vals[-1]], z0, t_eval=t_vals)


plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label="Predator Population (x)")
plt.plot(sol.t, sol.y[1], label="Prey Population (y)")
plt.xlabel("Time")
plt.ylabel("Population (thousands)")
plt.legend()
plt.title("Lotka-Volterra Predator-Prey Model")
plt.grid()
plt.show()

diff = np.abs(sol.y[0] - sol.y[1])
t_equal = sol.t[np.argmin(diff)]
print("First time when populations are equal:", t_equal)
