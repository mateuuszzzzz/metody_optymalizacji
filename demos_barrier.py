from matplotlib import pyplot as plt
import numpy as np

def show_barrier_function():
    def constraint1(x):
        return (-1-x)
    
    def constraint2(x):
        return (x-1)
    
    x = np.linspace(-2,2,1000)

    plt.plot(x, -(1/constraint1(x)+1/constraint2(x)), label='Bariera 1/x')
    plt.plot(x, -(np.log(-constraint1(x))+np.log(-constraint2(x))), label='Bariera log')
    
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.ylim(-0.5, 20)
    plt.xlim(-2, 2)
    plt.legend()
    plt.show()