#Import the numpy library
import numpy as np 
X=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])
y=np.array([[1],[1],[0]])
#Sigmoid Function
def sigmoid (x):
	return 1/(1 + np.exp(-x))

def derivatives_sigmoid(x):
	return x*(1-x)

#iterations
iterations=2000

#learning rate
neta=0.1

#number of neurons in input layer
I1=X.shape[1]

#number of neurons in hidden layers
H1=2
H2=2

#number of neurons in output layer
O1=1

#Random Weight and bias intialisation
W1=np.random.uniform(size=(I1,H1))
b1=np.random.uniform(size=(1,H1))
W2=np.random.uniform(size=(H1,H2))
b2=np.random.uniform(size=(1,H2))
W3=np.random.uniform(size=(H2, O1))
b3=np.random.uniform(size=(1,O1))
for x in range(iterations):
	
	#Forward Pass
	H1_input=np.dot(X,W1)
	H1_activations=sigmoid(H1_input+b1)

	H2_input=np.dot(H1_activations,W2)
	H2_activations=sigmoid(H2_input+b2)

	O1_input=np.dot(H2_activations,W3)
	O1_activations=sigmoid(O1_input+b3)

	#Backward Pass
	E=y-O1_activations
	slope_output_layer = derivatives_sigmoid(O1_activations)
	slope_hidden_layer = derivatives_sigmoid(H2_activations)
	d_output = E * slope_output_layer
	#print(d_output)
	Error_at_hidden_layer = d_output.dot(W3.T)
	d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
	slope_output_layer1 = derivatives_sigmoid(H2_activations)
	slope_hidden_layer2 = derivatives_sigmoid(H1_activations)
	d_output1 = E * slope_output_layer1
	Error_at_hidden_layer = d_output1.dot(W2.T)
	d_hiddenlayer1 = Error_at_hidden_layer * slope_hidden_layer2
	W3 += H2_activations.T.dot(d_output) *neta
	b3 += np.sum(d_output, axis=0,keepdims=True) *neta
	W2 += H1_activations.T.dot(d_hiddenlayer) *neta
	b2 += np.sum(d_hiddenlayer, axis=0,keepdims=True) *neta
	W1 += X.T.dot(d_hiddenlayer1) *neta
	b1 += np.sum(d_hiddenlayer1, axis=0,keepdims=True) *neta
	print (E)
