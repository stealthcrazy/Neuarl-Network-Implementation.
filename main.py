import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



df = pd.read_csv('o.csv') # loading the csv file

trainData = df.to_numpy() # converting the dataframe into a numpy array

# setting up the label
imgd = trainData[0][1:].reshape(28,28)
imgd = np.rot90(np.flip(imgd,1))
lbl = trainData[0][0]

