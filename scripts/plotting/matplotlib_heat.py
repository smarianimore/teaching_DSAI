import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    df = pd.read_csv("../../data/eurodist.csv", index_col="city")
    M = np.asarray(df)
    plt.imshow(M)
    plt.show()
