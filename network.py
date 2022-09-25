#!/usr/bin/env python3
#! coding: utf-8
#! python3
# Lame Neuronal Network (network)
# TDT 2022
# Thx The Independent Code https://www.youtube.com/watch?v=pauPCy_s0Ok


# Infers a network
def infer( network, input ):
	output = input
	for layer in network:
		output = layer.forward( output )
	return output

# Trains a network
def train( network, x_train, y_train, loss, loss_prime, epochs=1000, learning_rate=0.01, verbose=False ):
	for e in range( epochs ):
		error = 0
		for x, y in zip(x_train, y_train):
			
			# Forward: The inference part
			# The output of a layer is the input of the next layer
			output = infer( network, x )
			
			# Error
			# Mostly for display
			if verbose : error += loss( y, output )
			
			# Backward: The learning part
			# The input of a layer is the output of the previous one
			# but as we scan the network backwards, first inverse the output of the network
			gradient = loss_prime( y, output )
			for layer in reversed( network ):
				gradient = layer.backward( gradient, learning_rate )
			
		if verbose :
			error /= len( x_train )
			print( f"{e+1}/{epochs} error={error}" )
		
