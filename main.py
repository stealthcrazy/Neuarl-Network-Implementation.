import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Network
from Network import sig , SSR ,Manual_BackProp , GradientUpdate , SumGradients
# setting numpy seed to ensure same results
#np.random.seed(0) 

df = pd.read_csv('train.csv') # loading the csv file

trainData = df.to_numpy() # converting the dataframe into a numpy array

#shuffling the training data for data intialisation
rand = np.random.default_rng() 
rand.shuffle(trainData)

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




# forward propogation of Network Test
A2 = Test.forwardPropogate(Input)
C = SSR(A2,Label)


n = 0.1  #learning rate or step size for gradient descent
Gradients = None
count = 1
batch_size = 64 #batch size for mini batch for gradient descent
epochs = 20# epochs
epoch_size = 239999*0.98 # limit of the samples in epoch

Costs_data = []
Accuracy_data = []

#training the model using gradient descent
for i in range(epochs): 
    # setting accuracy counter
    acc = 0
    Costs = 0
    #suffle each epoch
    rand.shuffle(trainData)
    while count <= epoch_size:
        
        #setting batch accuracy
        batch_acc = 0
        
        for k in range(batch_size):

            #getting the data input and label
            Input = trainData[count][1:].reshape(784,1)
            Label = np.zeros((10,1))
            lbl = trainData[count][0]
            Label[lbl][0] = 1

            #forward propogation of model
            A2 = Test.forwardPropogate(Input)
            C = SSR(A2,Label)
            Costs+=C
            

            #accuracy update
            #print(np.argmax(A2) , lbl , f"---- count {count}" )
            if np.argmax(A2) == lbl:
                acc+=1
                batch_acc +=1

            #backpropogating model
            if Gradients !=None:
                #summing gradients of each sample in mini batch that affects the cost
                Gradients = SumGradients(Gradients, Manual_BackProp(Test,Input,Label))
            else:
                Gradients = Manual_BackProp(Test,Input,Label)
            
            

            
            count+=1
        GradientUpdate(Test , Gradients , n/batch_size) # gradient descent by updating the model
        Gradients = None # resets gradients for next calculation in descent
        #print(C , f"--- count {count} ----epoch {i} --- batch_accuracy => {batch_acc/batch_size}" , np.argmax(A2) , lbl)
        

    Costs_data.append(Costs/count) 
    count = 1
    print(C , f"--- epoch {i} --- accuracy => {acc/epoch_size}" , np.argmax(A2) , lbl)
    Accuracy_data.append(acc/epoch_size)
print(Costs_data)
print(Accuracy_data)
                
np.save("LayerWeights.npy",Test.LayerWeights_dict)
        
np.save("LayerBias.npy",Test.LayerBias_dict)

    
    
    
        