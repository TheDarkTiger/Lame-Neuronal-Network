#!/usr/bin/env python3
#! coding: utf-8
#! python3
# Lame Neuronal Network (activation layer)
# TDT 2022
# Thx The Independent Code https://www.youtube.com/watch?v=pauPCy_s0Ok

import numpy as np
from layer import Layer

# 
class Activation( Layer ):
	def __init__( self, activation, activation_prime ):
		self.activation = activation
		self.activation_prime = activation_prime
	
	#-------------------------------------------------------------------------
	# Forward function
	#
	# Y = f( X )
	#
	# @param input   Input to process
	#
	# @return Processed input (output)
	#-------------------------------------------------------------------------
	def forward( self, input ):
		self.input = input
		return self.activation( self.input )
	
	#-------------------------------------------------------------------------
	# Backward function
	#
	# Compute the gradient of the error with respect to the input
	# dX = dY (.) f'( X )
	#
	# @param output_gradient   Derivative of the error with respect to the the output
	# @param learning_rate     To pass an optimizer to optimise the gradient descent (speed up the learning) NOT USED HERE
	#
	# @return Derivative of the error with respect to the the input of the layer
	#-------------------------------------------------------------------------
	def backward( self, output_gradient, learning_rate ):
		return np.multiply( output_gradient, self.activation_prime( self.input ) )

