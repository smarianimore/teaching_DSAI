import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

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

fig = px.bar(medals_italy, x="Year", y="size", color="Medal", color_discrete_sequence=['#CD7F32','#DAA520','#C0C0C0'])
fig.show()
