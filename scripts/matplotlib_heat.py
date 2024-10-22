import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../data/eurodist.csv", index_col="city")
M = np.asarray(df)
plt.imshow(M)
plt.show()
