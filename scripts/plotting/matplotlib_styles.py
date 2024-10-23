import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    x = np.linspace(0, 2, 100)

    fig, ax = plt.subplots()
    ax.plot(x, x, 'd', label='linear', color='blue', linewidth=2)  # 'd' = diamond marker (== 'marker' param)
    ax.plot(x, x**2, label='quadratic', color='red', linewidth=2, marker='s')
    ax.plot(x, x**3, label='cubic', color='black', linewidth=2, linestyle=':')
    ax.set_xlabel('x')
    ax.set_ylabel('function of x')
    ax.set_title('Stylish plot (OO style)')
    ax.legend()
    plt.show()
