# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 10:10:46 2017

@author: Deepti
"""

import numpy as np

a = np.array([[0,-1],[2,3]])
eigenvalues,eigenvectors=np.linalg.eig(a)
print "Eigen Values: ", eigenvalues
print "Eigen Vetors: ", eigenvectors

###############EIGENVALUES AND EIGENVECTORS##########################
#Eigen Values:  [ 1.  2.], which are same the those I got on paper.
#Eigen Vetors:  [[-0.70710678  0.4472136 ]
# [ 0.70710678 -0.89442719]]

#EIGEN VECTORS ARE MULTIPLES OF WHAT I GOT ON PAPER SOLVING MANUALLY.