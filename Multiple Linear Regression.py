import pandas as pd
import os

############################## Data import ##############################
df = pd.read_csv(os.path.abspath('example_dataset_final.csv'),header=None)

X = df.iloc[2:, 1:5].values
y = df.iloc[2:, 5].values

print('a')