import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # WATCH OUT: undocumented dependecy on scipy

if __name__ == '__main__':

    df = pd.read_csv("../../data/eurodist.csv", index_col="city")
    sns.clustermap(df)
    plt.show()
