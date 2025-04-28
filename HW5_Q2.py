import numpy as np


# exact solution
def exact_solution(t):
    u1 = 2*np.exp(-3*t) - np.exp(-39*t) + (1/3)*np.cos(t)
    u2 = -np.exp(-3*t) + 2*np.exp(-39*t) - (1/3)*np.cos(t)
    return u1, u2

# system condition
def system(t, u):
    u1, u2 = u
    du1_dt = 9*u1 + 24*u2 + 5*np.cos(t) - (1/3)*np.sin(t)
    du2_dt = -24*u1 - 52*u2 - 9*np.cos(t) + (1/3)*np.sin(t)
    return np.array([du1_dt, du2_dt])

# Runge-Kutta method
def runge_kutta(system, t0, u0, h, t_end):
    t_values = np.arange(t0, t_end + h/2, h)
    n = len(t_values)
    
    u_values = np.zeros((n, len(u0)))
    u_values[0] = u0
    
    for i in range(1, n):
        t = t_values[i-1]
        u = u_values[i-1]
        
        k1 = system(t, u)
        k2 = system(t + h/2, u + h*k1/2)
        k3 = system(t + h/2, u + h*k2/2)
        k4 = system(t + h, u + h*k3)
        
        u_values[i] = u + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    return t_values, u_values

# initiate values
t0 = 0.0
u0 = np.array([4/3, 2/3])  # [u1(0), u2(0)]
t_end = 1.0

# h=0.05的Runge-Kutta方法
h1 = 0.05
t_rk1, u_rk1 = runge_kutta(system, t0, u0, h1, t_end)

# h=0.1的Runge-Kutta方法
h2 = 0.1
t_rk2, u_rk2 = runge_kutta(system, t0, u0, h2, t_end)


# 在Runge-Kutta方法的離散點上計算exact solution
u1_exact_rk1 = []
u2_exact_rk1 = []
for t in t_rk1:
    u1, u2 = exact_solution(t)
    u1_exact_rk1.append(u1)
    u2_exact_rk1.append(u2)
    
u1_exact_rk2 = []
u2_exact_rk2 = []
for t in t_rk2:
    u1, u2 = exact_solution(t)
    u1_exact_rk2.append(u1)
    u2_exact_rk2.append(u2)

u1_exact_rk1 = np.array(u1_exact_rk1)
u2_exact_rk1 = np.array(u2_exact_rk1)
u1_exact_rk2 = np.array(u1_exact_rk2)
u2_exact_rk2 = np.array(u2_exact_rk2)

# 計算error
error_u1_rk1 = np.abs(u_rk1[:, 0] - u1_exact_rk1)
error_u2_rk1 = np.abs(u_rk1[:, 1] - u2_exact_rk1)
error_u1_rk2 = np.abs(u_rk2[:, 0] - u1_exact_rk2)
error_u2_rk2 = np.abs(u_rk2[:, 1] - u2_exact_rk2)

# comaparing the results
print("\ncomparing the results (h=0.05)：")
print("t\tu1(RK)\t\tu1(Exact)\tError\t\tu2(RK)\t\tu2(Exact)\tError")
for i in range(len(t_rk1)):
    print(f"{t_rk1[i]:.2f}\t{u_rk1[i,0]:.8f}\t{u1_exact_rk1[i]:.8f}\t{error_u1_rk1[i]:.8f}\t{u_rk1[i,1]:.8f}\t{u2_exact_rk1[i]:.8f}\t{error_u2_rk1[i]:.8f}")

print("\ncomparing the results (h=0.1)：")
print("t\tu1(RK)\t\tu1(Exact)\tError\t\tu2(RK)\t\tu2(Exact)\tError")
for i in range(len(t_rk2)):
    print(f"{t_rk2[i]:.2f}\t{u_rk2[i,0]:.8f}\t{u1_exact_rk2[i]:.8f}\t{error_u1_rk2[i]:.8f}\t{u_rk2[i,1]:.8f}\t{u2_exact_rk2[i]:.8f}\t{error_u2_rk2[i]:.8f}")

