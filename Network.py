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
        self.LayerWeights_dict = {}
        self.LayerBias_dict = {}
        

        # setting up the weights and biases for each layer 
        #size of each weight is col size = previous layer node size , row size = current layer node size
        #using normal dist to intialise weights
        # storing the weights in a dictionary

        for i in range(hiddenLayerCount +1):
            if i != hiddenLayerCount :
                self.LayerBias_dict[f"hidden_{i}_bias"] = np.zeros((self.hiddenLayerNodes,1))
                if i == 0:
                    self.LayerWeights_dict[f"hidden_{i}_weight"] = np.random.normal(0.0,1,(self.hiddenLayerNodes,self.inputs))
                    
                else:
                    self.LayerWeights_dict[f"hidden_{i}_weight"] = np.random.normal(0.0,1,(self.hiddenLayerNodes , self.hiddenLayerNodes))
            
            else:
                self.LayerBias_dict[f"output_0_bias"] = np.zeros((self.outputs,1))
                self.LayerWeights_dict[f"output_0_weight"] = np.random.normal(0.0,1,(self.outputs,self.hiddenLayerNodes,))

        # setting up the stages manually *will implement automatic generation in future




    def forwardPropogate(self,Input):

        # forward propogation of Network manually *will implement automatic generation in future

        self.Z0 = (self.LayerWeights_dict["hidden_0_weight"]@Input) + self.LayerBias_dict["hidden_0_bias"]
        self.A0 = sig(self.Z0)
        self.Z1 = (self.LayerWeights_dict["hidden_1_weight"]@self.A0) + self.LayerBias_dict["hidden_1_bias"]
        self.A1 = sig(self.Z1)
        self.Z2 = (self.LayerWeights_dict["output_0_weight"]@self.A1) + self.LayerBias_dict["output_0_bias"]
        self.A2 = sig(self.Z2)

        return self.A2
        


# defining the sum of square residuals to calculate cost
def SSR(o , x ):

    return np.linalg.norm(o-x)**2


# defining the sigmoid
def sig(M , derivative = False):
    if derivative == True:
        return sig(M)*(1-sig(M))
    return 1/(np.exp(-M)+1)
       


def Manual_BackProp(Test,Input,Label):
    #storing manual backpropogation gradients That will be returned  *will implement automatic generation in future
    Gradients = {}
    


    # calculating garadients of last layer weights and biases

    dC_dA2 = 2*(Test.A2 - Label)
    dA2_dZ2 = sig(Test.Z2,derivative=True)
    S2 = np.multiply(dC_dA2,dA2_dZ2)

    dC_dW2 = S2@(Test.A1.T)
    dC_dB2 = S2

    #appending the gradients
    Gradients["output_0_weight_gradients"] = dC_dW2
    Gradients["output_0_bias_gradients"] = dC_dB2

    #print("___")
    #print(dC_dW2.shape , "dC_dW2")
    #print(dC_dB2.shape , "dC_dB2")
    #print("___")

    # calculating garadients of 2nd to last layer weights and biases
    W2 = Test.LayerWeights_dict["output_0_weight"]
    dC_dA1 = (W2.T)@S2
    dA1_dZ1 = sig(Test.Z1,derivative=True)
    S1 = np.multiply(dC_dA1,dA1_dZ1)

    dC_dW1 = S1@(Test.A0.T)
    dC_dB1 = S1

    #appending the gradients
    Gradients["hidden_1_weight_gradients"] = dC_dW1
    Gradients["hidden_1_bias_gradients"] = dC_dB1

    #print("___")
    #print(dC_dW1.shape , "dC_dW1")
    #print(dC_dB1.shape , "dC_dB1")
    #print("___")

    # calculating garadients of 3rd to last layer weights and biases
    W1 = Test.LayerWeights_dict["hidden_1_weight"]
    dC_dA0 = (W1.T)@S1
    dA0_dZ0 = sig(Test.Z0,derivative=True)
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

def SumGradients(CurrGrad , NewGrad):
    # manually summing the gradients for the minibatches
    CurrGrad["hidden_0_weight_gradients"] += NewGrad["hidden_0_weight_gradients"]
    CurrGrad["hidden_0_bias_gradients"] += NewGrad["hidden_0_bias_gradients"]

    CurrGrad["hidden_1_weight_gradients"] += NewGrad["hidden_1_weight_gradients"]
    CurrGrad["hidden_1_bias_gradients"] += NewGrad["hidden_1_bias_gradients"]

    CurrGrad["output_0_weight_gradients"] += NewGrad["output_0_weight_gradients"]
    CurrGrad["output_0_bias_gradients"] += NewGrad["output_0_bias_gradients"]

    return CurrGrad

def GradientUpdate(Test , Gradients , n):

    # manually adjusting gradients for weights and biases

    Test.LayerWeights_dict["hidden_0_weight"] = Test.LayerWeights_dict["hidden_0_weight"] - n*(Gradients["hidden_0_weight_gradients"])
    Test.LayerBias_dict["hidden_0_bias"] = Test.LayerBias_dict["hidden_0_bias"] - n*(Gradients["hidden_0_bias_gradients"])

    Test.LayerWeights_dict["hidden_1_weight"] = Test.LayerWeights_dict["hidden_1_weight"] - n*(Gradients["hidden_1_weight_gradients"])
    Test.LayerBias_dict["hidden_1_bias"] = Test.LayerBias_dict["hidden_1_bias"] - n*(Gradients["hidden_1_bias_gradients"])

    Test.LayerWeights_dict["output_0_weight"] = Test.LayerWeights_dict["output_0_weight"] - n*(Gradients["output_0_weight_gradients"])
    Test.LayerBias_dict["output_0_bias"] = Test.LayerBias_dict["output_0_bias"] - n*(Gradients["output_0_bias_gradients"])