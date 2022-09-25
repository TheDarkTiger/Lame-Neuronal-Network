#!/usr/bin/env python3
#! coding: utf-8
#! python3
# Lame Neuronal Network (losses functions)
# TDT 2022
# Thx The Independent Code https://www.youtube.com/watch?v=pauPCy_s0Ok

import numpy as np


# Mean Square Error loss function
# y is predicted output and y* is ground truth
# E = (1/n) sigma(i)( y* - y )²
def mse( true_y, predicted_y ):
	return np.mean( np.power( true_y - predicted_y, 2 ) )

# Mean Square Error loss function prime
# Y is predicted outputs and Y* is ground truths
# E' = (2/n) ( Y - Y*)
def mse_prime( true_y, predicted_y ):
	return 2 * (predicted_y - true_y) / np.size( true_y )
