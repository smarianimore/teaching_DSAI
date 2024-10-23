import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    x = np.linspace(0, 2, 100)

    fig, ax = plt.subplots()
    ax.plot(x, x, label='linear')
    ax.plot(x, x**2, label='quadratic')
    ax.plot(x, x**3, label='cubic')
    ax.set_yscale('log')
    ax.set_xlabel('x')
    ax.set_ylabel('function of x')
    ax.set_title('Simple plot (OO style)')
    ax.legend()
    plt.show()
