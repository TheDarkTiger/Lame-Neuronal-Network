#!/usr/bin/env python3
#! coding: utf-8
#! python3
# Lame Neuronal Network (layer)
# TDT 2022
# Thx The Independent Code https://www.youtube.com/watch?v=pauPCy_s0Ok

# 
class Layer:
	def __init__( self ):
		self.input = None
		self.output = None
	
	#-------------------------------------------------------------------------
	# Forward function
	#
	# @param input   Input to process
	#
	# @return Processed input (output)
	#-------------------------------------------------------------------------
	def forward( self, input ):
		# TODO return output
		pass
	
	#-------------------------------------------------------------------------
	# Backward function
	#
	# @param output_gradient   Derivative of the error with respect to the the output
	# @param learning_rate     To pass an optimizer to optimise the gradient descent (speed up the learning)
	#
	# @return Derivative of the error with respect to the the input of the layer
	#-------------------------------------------------------------------------
	def backward( self, output_gradient, learning_rate ):
		# TODO update parameter and return input gradient
		pass

