from matplotlib import pyplot as plt
import numpy as np

def show_max_penalty_function():
    def constraint(x):
        return (x-0.1)*(x-0.2)
    
    def penalty_with_square(x):
        return np.power(np.where(x<0, 0, x),2)

    x = np.linspace(-1,1,1000)
    y = constraint(x)

    plt.plot(x, y, label='g(x)')
    plt.plot(x,penalty_with_square(y), label='||max{0, g(x)}||^2')
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.ylim(-0.005, 0.05)
    plt.xlim(-0.25, 0.5)
    plt.legend()
    plt.show()

def show_external_penalty_function_method():
    def f(x):
        return x**2 - 10*x
    
    def constraint(x):
        return x - 3 # x - 3 <= 0
    
    def f_with_penalty(x,rho):
        return f(x) + rho * np.power(np.where(constraint(x)<0, 0, constraint(x)),2)

    x = np.linspace(-1,11,10000)

    for penalty in [0,1,2,4,8]:
        plt.plot(x, f_with_penalty(x,penalty), label=f'f(x) + {penalty}(max(0, g(x)))^2')
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.axvline(x=3, color='cyan', linestyle='--')
    plt.axvline(x=5, color='black', linestyle='--')
    plt.ylim(-50, 20)
    plt.legend()
    plt.show() 