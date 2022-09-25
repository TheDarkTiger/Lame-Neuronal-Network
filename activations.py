#!/usr/bin/env python3
#! coding: utf-8
#! python3
# Lame Neuronal Network (activations functions)
# TDT 2022
# Thx The Independent Code https://www.youtube.com/watch?v=pauPCy_s0Ok

import numpy as np
from activation import Activation


# tanh() activation function and it's prime
# f = tanh( x )
# f' = 1 - (tanh( x )²)
class Tanh( Activation ):
	def __init__( self ):
		tanh = lambda x: np.tanh( x )
		tanh_prime = lambda x: 1 - ( np.tanh( x )**2 )
		super().__init__( tanh, tanh_prime )
