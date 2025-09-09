import numpy as np




class NeuralNetwork:

    def __init__(
            self,
            inputs,
            hiddenLayerCount,
            hiddenLayerNodes,
            outputs,

                 ):
        # intialsing the network parameters such as inputs , hidden layers and outputs
        self.inputs = inputs
        self.hiddenLayerCount = hiddenLayerCount
        self.hiddenLayerNodes = hiddenLayerNodes
        self.outputs = outputs
        

        # intialising the dictionaries to store weights and biases for easy access and tracking
        self.LayerWights_dict = {}
        self.LayerBias_dict = {}
        

        # setting up the weights and biases for each layer 
        #size of each weight is col size = previous layer node size , row size = current layer node size
        #using normal dist to intialise weights
        # storing the weights in a dictionary

        for i in range(hiddenLayerCount +1):
            if i != hiddenLayerCount :
                self.LayerBias_dict[f"hidden_{i}_bias"] = np.zeros((self.hiddenLayerNodes,1))
                if i == 0:
                    self.LayerWights_dict[f"hidden_{i}_weight"] = np.random.normal(0.0,0.5,(self.hiddenLayerNodes,self.inputs))
                    
                else:
                    self.LayerWights_dict[f"hidden_{i}_weight"] = np.random.normal(0.0,0.5,(self.hiddenLayerNodes , self.hiddenLayerNodes))
            
            else:
                self.LayerBias_dict[f"output_0_bias"] = np.zeros((self.outputs,1))
                self.LayerWights_dict[f"output_0_weight"] = np.random.normal(0.0,0.5,(self.outputs,self.hiddenLayerNodes,))


# defining the sum of square residuals to calculate cost
def SSR(o , x ):

    return np.linalg.norm(o-x)**2


# defining the sigmoid
def sig(M , derivative = False):
    if derivative == True:
        return sig(M)*(1-sig(M))
    return 1/(np.exp(-M)+1)
       


def Manual_BackProp(Z0,A0,Z1,A1,Z2,A2,Test,Input,Label):
    #storing manual backpropogation gradients That will be returned
    Gradients = {}
    


    # calculating garadients of last layer weights and biases

    dC_dA2 = 2*(A2 - Label)
    dA2_dZ2 = sig(Z2,derivative=True)
    S2 = np.multiply(dC_dA2,dA2_dZ2)

    dC_dW2 = S2@(A1.T)
    dC_dB2 = S2

    #appending the gradients
    Gradients["output_0_weight_gradients"] = dC_dW2
    Gradients["output_0_bias_gradients"] = dC_dB2

    #print("___")
    #print(dC_dW2.shape , "dC_dW2")
    #print(dC_dB2.shape , "dC_dB2")
    #print("___")

    # calculating garadients of 2nd to last layer weights and biases
    W2 = Test.LayerWights_dict["output_0_weight"]
    dC_dA1 = (W2.T)@S2
    dA1_dZ1 = sig(Z1,derivative=True)
    S1 = np.multiply(dC_dA1,dA1_dZ1)

    dC_dW1 = S1@(A0.T)
    dC_dB1 = S1

    #appending the gradients
    Gradients["hidden_1_weight_gradients"] = dC_dW1
    Gradients["hidden_1_bias_gradients"] = dC_dB1

    #print("___")
    #print(dC_dW1.shape , "dC_dW1")
    #print(dC_dB1.shape , "dC_dB1")
    #print("___")

    # calculating garadients of 3rd to last layer weights and biases
    W1 = Test.LayerWights_dict["hidden_1_weight"]
    dC_dA0 = (W1.T)@S1
    dA0_dZ0 = sig(Z0,derivative=True)
    S0 = np.multiply(dC_dA0,dA0_dZ0)

    dC_dW0 = S0@(Input.T)
    dC_dB0 = S0

    #appending the gradients
    Gradients["hidden_0_weight_gradients"] = dC_dW0
    Gradients["hidden_0_bias_gradients"] = dC_dB0


    #print("___")
    #print(dC_dW0.shape , "dC_dW0")
    #print(dC_dB0.shape , "dC_dB0")
    #print("___")

    return Gradients