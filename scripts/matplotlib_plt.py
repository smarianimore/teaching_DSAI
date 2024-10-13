import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = np.linspace(0, 2, 100)

    plt.figure()
    plt.plot(x, x, label='linear')
    plt.plot(x, x**2, label='quadratic')
    plt.plot(x, x**3, label='cubic')
    plt.xlabel('x')
    plt.ylabel('function of x')
    plt.title('Simple plot (Pyplot style)')
    plt.legend()
    plt.show()
