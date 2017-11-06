import pandas as pd
import os

csvfiles = os.listdir(r'../data/csv')

for csv in csvfiles:
    pd.read_csv(csv)