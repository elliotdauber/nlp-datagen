import pandas as pd

df = pandas.read_csv('generated_data/gospel.csv')
print(len(df.index))