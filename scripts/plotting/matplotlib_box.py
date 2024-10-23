import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':

    df = pd.read_csv("../../data/mtcars.csv")

    df_4_cyl = df[df["cyl"] == 4]["mpg"]
    df_6_cyl = df[df["cyl"] == 6]["mpg"]
    df_8_cyl = df[df["cyl"] == 8]["mpg"]

    box_data = [df_4_cyl, df_6_cyl, df_8_cyl]
    box = plt.boxplot(box_data, tick_labels=["4 cyl", "6 cyl", "8 cyl"])

    plt.show()
