import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    M = 2
    N = 100

    sampled_mean = []
    for i in range(N):
        dice = []
        for j in range(M):
            dice.append(np.random.randint(1,6))
        sampled_mean.append(sum(dice)/len(dice))

    plt.hist(sampled_mean, np.arange(1.0, 6.0, 0.1), align='left')
    plt.show()
