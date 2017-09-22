# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 09:43:42 2017

@author: Deepti
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read the csv file and store it in a data frame
csv_reader = pd.read_csv("A:\\Fall 2017\\ML\\dataset_1.csv")
#csv_reader = pd.read_csv("dataset_1.csv")
df = pd.DataFrame(csv_reader)

#Convert the dataframe into a numpy array
data_matrix = df.as_matrix()

#Calculate variance
variance_of_x = np.var(data_matrix[:,0])
print "Variance of x: ",variance_of_x

variance_of_y = np.var(data_matrix[:,1])
print "Variance of y: ",variance_of_y

variance_of_z = np.var(data_matrix[:,2])
print "Variance of z: ",variance_of_z

#To calculate covariance, we take transpose of our data, as cov function expects one 
# random variable to be a row and our data has one random variable as column
covariance_xy = np.cov(data_matrix[:,0].T,data_matrix[:,1].T)
print "Covariance of x and y: ", covariance_xy

covariance_yz = np.cov(data_matrix[:,1].T,data_matrix[:,2].T)
print "Covariance of y and z: ", covariance_yz

#Calculating the mean centered matrix
mean_val=data_matrix.mean(axis=0)
mean_centered =(data_matrix - mean_val)

#Calculating the covariance matrix. Using transpose of mean_centered matrix, 
#as cov function expects a random variable to be a row, but our mean_centered 
#matrix has them as columns. Thus, transposing it.
covariance_matrix=np.cov(mean_centered.T)

#Finding eigenvalues and eigenvectors of covariance matrix
[eigenvalues,eigenvectors] = np.linalg.eig(covariance_matrix)
print "Eigen Values: ", eigenvalues
print "Eigen Vetors: ", eigenvectors

#Sorting eigenvalues and eigenvectors in descending order
index = np.argsort(eigenvalues)
index = index[::-1]
eigenvectors = eigenvectors[:,index]
eigenvalues = eigenvalues[index]
print "Eigen Values: ", eigenvalues
print "Eigen Vetors: ", eigenvectors

#Calculating the reduced matrix Y
Y=np.dot(data_matrix,eigenvectors)
print "Y : ", Y

#Plotting PC1 and PC2
plt.plot(Y[:,0],Y[:,1],'ro')
plt.show()

##################### SOLUTION TO VARIANCE, COVARIANCE AND PCA#################

#Variance of x:  0.080529305884
#Variance of y:  2.09690259152
#Variance of z:  0.080501954879

#Covariance of x and y:  [[ 0.08060992  0.40242878]
# [ 0.40242878  2.09900159]]
#Covariance of y and z:  [[ 2.09900159 -0.01439466]
# [-0.01439466  0.08058254]]

# PCA:

#Eigen Values:  [ 2.17638133  0.00333122  0.0804815 ]
#Eigen Vetors:  
# [[ 0.18857784  0.982048    0.00448705]
# [ 0.98203351 -0.18860355  0.00623651]
# [-0.00697082 -0.00323037  0.99997049]]

# Eigenvectors and values SORTED:
    
#Eigen Values:  [ 2.17638133  0.0804815   0.00333122]
#Eigen Vetors:  
# [[ 0.18857784  0.00448705  0.982048  ]
# [ 0.98203351  0.00623651 -0.18860355]
# [-0.00697082  0.99997049 -0.00323037]]

#Y matrix :(matrix of Scores)  
# [[ 0.80076073  0.52936936 -0.07064933]
# [ 1.66940231  0.54363259 -0.00636587]
# [ 2.61022858  0.08038336  0.0574392 ]
# ..., 
# [ 2.91800803  0.87258974  0.06905263]
# [-0.31050093  0.14209605  0.07934032]
# [ 4.08077688  0.3814206  -0.01146129]]

##############################################################################