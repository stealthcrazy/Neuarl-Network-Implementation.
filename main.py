import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Network
from Network import sig , SSR ,Manual_BackProp , GradientUpdate , SumGradients
# setting numpy seed to ensure same results
np.random.seed(0) 

df = pd.read_csv('train.csv') # loading the csv file

trainData = df.to_numpy() # converting the dataframe into a numpy array
print(trainData.shape)
# setting up the label
imgd = trainData[0][1:].reshape(28,28)
imgd = np.rot90(np.flip(imgd,1))
lbl = trainData[0][0]

Input = trainData[0][1:].reshape(784,1)
Label = np.zeros((10,1))
Label[lbl][0] = 1
print(Label , lbl)

Test = Network.NeuralNetwork(784 , 2 ,16 , 10 )


# forward propogation of Network


A2 = Test.forwardPropogate(Input)
C = SSR(A2,Label)
n = 0.001  #learning rate or step size for gradient descent
Gradients = None
count = 1
batch_size = 100
for i in range(10000):

    for j in range(batch_size):
        Input = trainData[count][1:].reshape(784,1)
        Label = np.zeros((10,1))
        lbl = trainData[count][0]
        Label[lbl][0] = 1

        A2 = Test.forwardPropogate(Input)
        C = SSR(A2,Label)
        
        if Gradients !=None:
            Gradients = SumGradients(Gradients, Manual_BackProp(Test,Input,Label))
        else:
            Gradients = Manual_BackProp(Test,Input,Label)
        count+=1
        if count ==239999:
            count = 0
        
        

    GradientUpdate(Test , Gradients , n/batch_size)
    Gradients = None
    print(C , f"--- count {count}")
        