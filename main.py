import numpy as np
import matplotlib.pyplot as np
import pandas as pd



df = pd.read_csv('train.csv') # loading the csv file

trainData = df.to_numpy() # converting the dataframe into a numpy array