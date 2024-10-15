import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':

    df = pd.read_csv("../data/planets.csv")
    df.columns = df.columns.str.strip()  # remove trailing whitespaces

    ax = sns.barplot(data=df, x="Name", y="SurfaceGravity", color="blue")
    ax.set(xlabel='', ylabel='Gravity [$m/s^2$]')
    plt.show()
