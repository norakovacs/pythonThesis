#!/usr/bin/env python
# coding: utf-8

#Plotting prediction vs test data for each curve, in xx,yy,xy dir

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  

# In[ ]:

mainDir = '/home/knora/pythonThesis/nora/'
testData = '/home/knora/data/MNNData/test_rnm1.data'

mnnout = []
test = []

lowest = 0


def readFile ( infilename ):
        
    infile = open(infilename, "r");
    tdata = infile.readlines();
    infile.close();
    
    sizetdata = len(tdata);
         
    for i in range(sizetdata):
        tdata[i] = tdata[i].split()
    
    #tdata = [x for x in tdata if x != []];
    #print(tdata)

    y = np.array([np.array(x) for x in tdata if x!=[]])
    
    smpplt = y.copy();
    smpplt = np.array(smpplt, dtype=np.float64)
    #print(smpplt[3000][0])
    return smpplt


test = readFile(testData)

for n in range(1):
	if n == lowest:
	    outfile = 'prnnoffline.out'
	    pred = os.path.join(mainDir, outfile)
	    mnnout = readFile(pred)
	    for j in range(54):
	    	pdf = os.path.join(mainDir,'Net%03i_%i.jpg' % (n,j))
	    	with plt.style.context('ggplot'):
	    	    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))
	    	    xy = ['$_{x}$', '$_{y}$', '$_{xy}$']
	    	    for k in range(3):
	            	axs[k].plot(mnnout[j*61:(j+1)*61,k], mnnout[j*61:(j+1)*61,k+3], 'm-', linewidth=0.5)
	            	axs[k].plot(test[j*61:(j+1)*61,k], test[j*61:(j+1)*61,k+3], 'g-', linewidth=0.5)
	            	axs[k].set_xlabel(r'$\epsilon$' + xy[k])
	            	axs[k].set_ylabel(r'$\sigma$' + xy[k])
	    	    fig.tight_layout()
	    	    fig.savefig(pdf, format='jpg', dpi=300)
	    	    plt.close(fig)
	
        
	
