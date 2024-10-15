import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':

    df = pd.read_csv("../data/tesla-stock-price.csv")
    timesteps = df["date"].astype("datetime64[ns]")
    prices = df["open"].astype("float64")

    ax = sns.lineplot(x=timesteps, y=prices, marker='o', markersize=5, linestyle='-')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()

    plt.show()
