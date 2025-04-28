import numpy as np




def exact_solution(t):
    return t * np.tan(np.log(t))


def f(t, y):
    return 1 + (y/t) + (y/t)**2

# Euler方法
def euler_method(f, t0, y0, h, t_end):
    t_values = np.arange(t0, t_end + h/2, h)
    y_values = np.zeros_like(t_values)
    y_values[0] = y0
    
    for i in range(1, len(t_values)):
        y_values[i] = y_values[i-1] + h * f(t_values[i-1], y_values[i-1])
    
    return t_values, y_values

# Taylor方法（二階）
def taylor_method(f, t0, y0, h, t_end):
    # 定義dy/dt的導數
    def df_dt(t, y):
        return -y/t**2 - 2*y**2/t**3
    
    # 定義df/dy
    def df_dy(t, y):
        return 1/t + 2*y/t**2
    
    t_values = np.arange(t0, t_end + h/2, h)
    y_values = np.zeros_like(t_values)
    y_values[0] = y0
    
    for i in range(1, len(t_values)):
        t, y = t_values[i-1], y_values[i-1]
        f_value = f(t, y)
        
        # 計算d²y/dt²
        d2y_dt2 = df_dt(t, y) + df_dy(t, y) * f_value
        
        # 二階Taylor展開
        y_values[i] = y + h * f_value + (h**2/2) * d2y_dt2
    
    return t_values, y_values

# 參數設置
t0 = 1.0  
y0 = 0.0  
h = 0.1   
t_end = 2.0  

# Euler方法
t_euler, y_euler = euler_method(f, t0, y0, h, t_end)

# Taylor方法
t_taylor, y_taylor = taylor_method(f, t0, y0, h, t_end)

# exact solution
t_exact = np.linspace(t0, t_end, 100)
y_exact = exact_solution(t_exact)

# 在Euler和Taylor方法的離散點上計算exact solution，以便比較
y_exact_euler = exact_solution(t_euler)
y_exact_taylor = exact_solution(t_taylor)

# 計算error
euler_error = np.abs(y_euler - y_exact_euler)
taylor_error = np.abs(y_taylor - y_exact_taylor)

print("comparing the results of Euler and Taylor：")
print("t\tEuler\t\tExact\t\tError\t\tTaylor\t\tError")
for i in range(len(t_euler)):
    print(f"{t_euler[i]:.1f}\t{y_euler[i]:.8f}\t{y_exact_euler[i]:.8f}\t{euler_error[i]:.8f}\t{y_taylor[i]:.8f}\t{taylor_error[i]:.8f}")

