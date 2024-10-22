import mplcursors
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/medals.csv")

medals_by_country_and_year = df.groupby(["Country", "Year"]).size()

medals_italy = df[df["Country"] == "ITA"]
medals_italy["Medal"] = medals_italy["Medal"].astype('category')

medals_italy = medals_italy.groupby(["Year", "Medal"], as_index=False).size()

gold = medals_italy[medals_italy["Medal"] == "Gold"]
silver = medals_italy[medals_italy["Medal"] == "Silver"]
bronze = medals_italy[medals_italy["Medal"] == "Bronze"]

fig, ax = plt.subplots()
ax.set_ylabel("Medals")
ax.bar(gold["Year"], gold["size"].values, 2, label='Gold', color='#DAA520')
ax.bar(silver["Year"], silver["size"].values, 2, bottom=gold["size"].values,
       label='Silver', color='#C0C0C0')
ax.bar(bronze["Year"], bronze["size"].values, 2,
       bottom=gold["size"].values+silver["size"].values,
       label='Bronze', color='#CD7F32')

cursor = mplcursors.cursor(hover=mplcursors.HoverMode.Transient)
@cursor.connect("add")
def on_add(sel):
    x, y, width, height = sel.artist[sel.index].get_bbox().bounds
    sel.annotation.set(text=f"{int(x+width/2)}: {int(height)}",
                       position=(0, 20), anncoords="offset points")
    sel.annotation.xy = (x + width / 2, y + height)
plt.show()

