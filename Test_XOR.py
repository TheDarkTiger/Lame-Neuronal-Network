#!/usr/bin/env python3
#! coding: utf-8
#! python3
# Lame Neuronal Network
# TDT 2022
# Thx The Independent Code https://www.youtube.com/watch?v=pauPCy_s0Ok

import numpy as np

from dense import Dense
from activations import Tanh
from losses import mse, mse_prime

print( "Hi there." )

# Desired inputs and output
print( "Set up inputs and outputs..." )
X = np.reshape( [ [0,0], [0,1], [1,0], [1,1] ], (4,2,1) )
Y = np.reshape( [ [0], [1], [1], [0] ], (4,1,1) )

# Define the network
# Layer 1 : 2 inputs
# Layer 2 : 3 hidden neurons
# Layer 3 : 1 output
print( "Set up the empty network..." )
network = [
	Dense( 2, 3 ),
	Tanh(),
	Dense( 3, 1 ),
	Tanh()
]

# Train parameters
print( "Set up training parameters..." )
epochs = 10000
learning_rate = 0.1

# train
print( "Starting training..." )
for e in range( epochs ):
	error = 0
	for x, y in zip(X, Y):
		
		# Forward: The inference part
		# The output of a layer is the input of the next layer
		output = x
		for layer in network:
			output = layer.forward( output )
		
		# Error
		# Mostly for display
		error += mse( y, output )
		
		# Backward: The learning part
		# The input of a layer is the output of the previous one
		# but as we scan the network backwards, first inverse the output of the network
		gradient = mse_prime( y, output )
		for layer in reversed( network ):
			gradient = layer.backward( gradient, learning_rate )
		
	error /= len( X )
	print( f"{e+1}/{epochs} error={error}" )















