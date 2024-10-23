import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':

    df = pd.read_csv("../../data/mtcars.csv")

    ax = sns.boxplot(data=df, x="cyl", y="mpg", palette='colorblind', hue=df["cyl"], legend=False)
    ax.set_xlabel('Number of cylinders')
    ax.set_ylabel('Miles per gallon')
    plt.show()
