import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Network
from Network import sig , SSR ,Manual_BackProp
# setting numpy seed to ensure same results
np.random.seed(0) 

df = pd.read_csv('o.csv') # loading the csv file

trainData = df.to_numpy() # converting the dataframe into a numpy array

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

Z0 = (Test.LayerWights_dict["hidden_0_weight"]@Input) + Test.LayerBias_dict["hidden_0_bias"]
A0 = sig(Z0)
Z1 = (Test.LayerWights_dict["hidden_1_weight"]@A0) + Test.LayerBias_dict["hidden_1_bias"]
A1 = sig(Z1)
Z2 = (Test.LayerWights_dict["output_0_weight"]@A1) + Test.LayerBias_dict["output_0_bias"]
A2 = sig(Z2)
C = SSR(A2,Label)


print(C)


# Manual Backpropogation as a Test

# calculating garadients of last layer weights and biases

dC_dA2 = 2*(A2 - Label)
dA2_dZ2 = sig(Z2,derivative=True)
S2 = np.multiply(dC_dA2,dA2_dZ2)

dC_dW2 = S2@(A1.T)
dC_dB2 = S2
print("___")
print(dC_dW2.shape , "dC_dW2")
print(dC_dB2.shape , "dC_dB2")
print("___")

# calculating garadients of 2nd to last layer weights and biases
W2 = Test.LayerWights_dict["output_0_weight"]
dC_dA1 = (W2.T)@S2
dA1_dZ1 = sig(Z1,derivative=True)
S1 = np.multiply(dC_dA1,dA1_dZ1)

dC_dW1 = S1@(A0.T)
dC_dB1 = S1
print("___")
print(dC_dW1.shape , "dC_dW1")
print(dC_dB1.shape , "dC_dB1")
print("___")

# calculating garadients of 3rd to last layer weights and biases
W1 = Test.LayerWights_dict["hidden_1_weight"]
dC_dA0 = (W1.T)@S1
dA0_dZ0 = sig(Z0,derivative=True)
S0 = np.multiply(dC_dA0,dA0_dZ0)

dC_dW0 = S0@(Input.T)
dC_dB0 = S0
print("___")
print(dC_dW0.shape , "dC_dW0")
print(dC_dB0.shape , "dC_dB0")
print("___")
for i in range(100):
    Gradients = Manual_BackProp(Z0,A0,Z1,A1,Z2,A2,Test,Input,Label)

    n = 0.1 # learning rate or step size for gradient descent

    # manually adjusting gradients for weights and biases

    Test.LayerWights_dict["hidden_0_weight"] = Test.LayerWights_dict["hidden_0_weight"] - n*(Gradients["hidden_0_weight_gradients"])
    Test.LayerBias_dict["hidden_0_bias"] = Test.LayerBias_dict["hidden_0_bias"] - n*(Gradients["hidden_0_bias_gradients"])

    Test.LayerWights_dict["hidden_1_weight"] = Test.LayerWights_dict["hidden_1_weight"] - n*(Gradients["hidden_1_weight_gradients"])
    Test.LayerBias_dict["hidden_1_bias"] = Test.LayerBias_dict["hidden_1_bias"] - n*(Gradients["hidden_1_bias_gradients"])

    Test.LayerWights_dict["output_0_weight"] = Test.LayerWights_dict["output_0_weight"] - n*(Gradients["output_0_weight_gradients"])
    Test.LayerBias_dict["output_0_bias"] = Test.LayerBias_dict["output_0_bias"] - n*(Gradients["output_0_bias_gradients"])

    Z0 = (Test.LayerWights_dict["hidden_0_weight"]@Input) + Test.LayerBias_dict["hidden_0_bias"]
    A0 = sig(Z0)
    Z1 = (Test.LayerWights_dict["hidden_1_weight"]@A0) + Test.LayerBias_dict["hidden_1_bias"]
    A1 = sig(Z1)
    Z2 = (Test.LayerWights_dict["output_0_weight"]@A1) + Test.LayerBias_dict["output_0_bias"]
    A2 = sig(Z2)
    C = SSR(A2,Label)

    print(C)