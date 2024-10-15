import matplotlib.pyplot as plt

values = [10, 12, 4, 16, 8]
countries = ["USA", "ITA", "FRA", "GBR", "GER"]
pct = [float(i)/sum(values)*100 for i in values]
plt.pie(values, labels = countries, autopct='%1.1f%%')
plt.show()
