import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/quakes.csv")
ax = sns.scatterplot(data=df, x="Latitude", y="Longitude", hue="Depth", size='Magnitude')
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
plt.show()
