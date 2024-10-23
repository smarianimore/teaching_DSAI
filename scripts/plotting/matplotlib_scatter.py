import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

if __name__ == '__main__':

    df = pd.read_csv("../../data/planets.csv")
    df.columns = df.columns.str.strip()  # remove trailing whitespaces
    df['Type'] = ['Rocky', 'Rocky', 'Rocky', 'Rocky', 'Gas', 'Gas', 'Gas', 'Gas']

    fig, ax = plt.subplots()
    colors = {'Rocky':'red', 'Gas':'blue'}
    scatter = ax.scatter(df["Diameter"], df["SurfaceGravity"], color=df['Type'].map(colors))
    ax.set_xscale('log')
    ax.set_xlabel("Diameter [$m$]")
    ax.set_ylabel("Gravity [$m/s^2$]")

    custom = [Line2D([], [], marker='.', color='red', linestyle='None'),
              Line2D([], [], marker='.', color='blue', linestyle='None')]
    plt.legend(handles=custom, labels=['Rocky', 'Gas'])

    plt.show()
