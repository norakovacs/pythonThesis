#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 18:17:41 2022

@author: malvesmaia
"""

# ------------------------ Importing libraries -------------------------------

import random
import numpy as np
import src 

# ------------------------ Auxiliary functions -------------------------------

def readTrData (infilename ):
    infile = open(infilename, "r")
    tdata = infile.readlines()
    infile.close()
           
    for i in range(len(tdata)):
        tdata[i] = tdata[i].split()
                    
    tdata = np.delete(tdata, 0, 1)
    tdata = np.delete(tdata, 0, 1)
     
    trdata = np.array(tdata[:-1], dtype=np.int64)
    loadtype = tdata[-1][0]

    return trdata, loadtype

# Reading file with information on the network and setting random seed 

parameters, loadtype = readTrData ('tData.txt')
rseed = int(parameters[7])
random.seed ( rseed )
parameters[7] = random.uniform(0, 100000) 

# --------------------------------- Main -------------------------------------

# File with training, validation and/or test sets

dataset = ['twonm1_061.data']

# Create NN 
 
src.PRNN(dataset, parameters, loadtype, 
         normalize = False, saveError = True, pretrained = False, 
               warmStart = False, evalncurves = 54)


# --------------------------- Extra comments ---------------------------------
# To train some NN from scratch, set both preTrained and warmStart to False.
#
# If you want to resume some interrupted training, set warmStart to True (and 
# keep preTrained as False). The name of the file used to recover weights 
# will be automatically recovered according to the trData.txt file.
#
# If you need to evaluate a test set (without training), the way to go is
# to set the preTrained to True and specify the number of curves you want to 
# evaluate. Otherwise, it will evaluate the entire file provided.
# No need to set warmStart, the name of the file with the optimal 
# parameters will be recovered automatically according to the trData.txt file.
#
# Finally, evalncurves is only necessary/read when preTrained is set to True and 
# the network is used in evaluation mode.
