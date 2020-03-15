import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

wine_data = pd.read_csv('winequality-white.csv', sep=';')

print(wine_data.describe())
