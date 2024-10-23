import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    df = pd.read_csv("../../data/planets.csv")
    df.columns = df.columns.str.strip()  # remove trailing whitespaces

    fig, ax = plt.subplots()
    ax.bar(df["Name"], df["SurfaceGravity"])
    ax.legend()
    ax.set_ylabel("Gravity [$m/s^2$]")
    plt.show()
