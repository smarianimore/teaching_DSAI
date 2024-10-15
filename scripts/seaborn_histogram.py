import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':

    M = 2
    N = 100

    sampled_mean = []
    for i in range(N):
        dice = []
        for j in range(M):
            dice.append(np.random.randint(1,6))
        sampled_mean.append(sum(dice)/len(dice))

    sns.histplot(sampled_mean, bins=np.arange(1, 6, .1))
    plt.show()
