#!/usr/bin/env python3
#! coding: utf-8
#! python3
# Lame Neuronal Network (dense layer)
# TDT 2022
# Thx The Independent Code https://www.youtube.com/watch?v=pauPCy_s0Ok

import numpy as np
from layer import Layer

# 
class Dense:
	def __init__( self, input_size, output_size ):
		self.weights = np.random.randn( output_size, input_size )
		self.bias = np.random.randn( output_size, 1 )
	
	#-------------------------------------------------------------------------
	# Forward function
	#
	# Simply computes Y = W.X + B
	# Y is the collumn vector ('list') of output, W the matrix of weight, X the collumn vector ('list') of input, and B the collumn vector ('list') of biases
	#
	# @param input   Input to process
	#
	# @return Processed input (output)
	#-------------------------------------------------------------------------
	def forward( self, input ):
		self.input = input
		return np.dot( self.weights, self.input ) + self.bias
	
	#-------------------------------------------------------------------------
	# Backward function
	#
	# Three things are computed here: The Weight gradient, the Bias gradient, and the Input gradient (whom is the output gratient of the previous layer)
	# dW = dY . Xt
	# dB = dY
	# dX = Wt . dY
	#
	# @param output_gradient   Derivative of the error with respect to the the output
	# @param learning_rate     To pass an optimizer to optimise the gradient descent (speed up the learning)
	#
	# @return Derivative of the error with respect to the the input of the layer
	#-------------------------------------------------------------------------
	def backward( self, output_gradient, learning_rate ):
		# dX
		input_gradient = np.dot( self.weights.T, output_gradient )
		
		# dW
		weight_gradient = np.dot( output_gradient, self.input.T )
		self.weights -= learning_rate * weight_gradient
		
		# dB
		self.bias -= learning_rate * output_gradient
		
		#return dX
		return input_gradient

