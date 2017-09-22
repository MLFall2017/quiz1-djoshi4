# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 09:08:48 2017

@author: Deepti
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read the csv file and store it in a data frame
csv_reader = pd.read_csv("A:\\Fall 2017\\ML\\dataset_1.csv")
df = pd.DataFrame(csv_reader)

#Convert the dataframe into a numpy array
data_matrix = df.as_matrix()

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

#Calculating the reduced matrix Y
Y=np.dot(data_matrix,eigenvectors)
print "Y : ", Y

#Plotting PC1 and PC2
plt.plot(Y[:,0],Y[:,1],'ro')
plt.show()