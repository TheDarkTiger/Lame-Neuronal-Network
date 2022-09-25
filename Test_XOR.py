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
from network import infer, train

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

# Train
print( "Starting training..." )
train( network, X, Y, mse, mse_prime, epochs = 10000, learning_rate = 0.1, verbose=True )

# Infer
print( "Infers to check results..." )

for test_case in [ [[0],[0]], [[0],[1]], [[1],[0]], [[1],[1]] ]:
	print( f"{test_case} :", end='' )
	print( infer( network, test_case) )







