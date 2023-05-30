#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 18:25:38 2022

@author: malvesmaia
"""

# Adapted:       Marina A. Maia
# Date:          Dec 2021

# ------------------------ Importing libraries -------------------------------

import os
import torch
import copy
import time
import numpy as np 
from numpy import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
from annmodel_1 import neural_network

# ----------------------------- GPU related ----------------------------------

is_cuda = torch.cuda.is_available()
    
if is_cuda:
  device = torch.device("cpu")
  print("GPU is available")
else:
  device = torch.device("cpu")
  print("GPU not available, CPU used")
  
# ------------------------------ Timer --------------------------------------
  
class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError("Timer is running. Use .stop() to stop it")
        self._start_time = time.clock()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")

        elapsed_time = time.clock() - self._start_time
        self._start_time = None
        return elapsed_time

# ------------------------------ Dataset -------------------------------------

class timeseries(Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float64).to(device)
        self.y = torch.tensor(y,dtype=torch.float64).to(device)
        self.len = x.shape[0]

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
  
    def __len__(self):
        return self.len


# ------------------------ Optimization problem -----------------------------    
    
class PRNN ( ):
  def __init__(self, trainingdata, parameters, loadtype, 
               normalize = False, writeEvery = 50, 
               saveError = True, pretrained = False, 
               warmStart = False, evalncurves = -1):
   
     self.trainingDataset = trainingdata
     self.bulkPts = int(parameters[0])        # number of bulk mat pts
     self.cohPts = int(parameters[1])         # number of coh mat pts
     self.maxIt = int(parameters[2])          # max number of epochs
     self.writeEvery = int(parameters[3])     # check val error every N epochs
     self.earlyStop = int(parameters[4])      # max number of epochs w/out improv. 
     self.trsize = int(parameters[5])
     self.skipFirst = int(parameters[6])
     self.rseed = int(parameters[7])
     self.batchSize = int(parameters[8])
     self.loadtype = loadtype
     self.normalize = normalize               # default is no normalization
     self.evalError = saveError               # calculate error at the end of tr
     self.preTrained = pretrained             # evaluation mode, skip training
     self.warmStart = warmStart               # resume training with a set of params
     self.evalncurves = evalncurves           # number of curves tested

     self.printTangent = False                # Print tangent stiffness 
                                              # of val/test set for debugging

     self.ls =  int(parameters[0]*3+parameters[1]*2)     # material layer size

     if self.preTrained:                      # if used for evaluation only,
       self.skipFirst = 0                     # no validation set is required
     
     # Setting random seed
     
     random.seed ( self.rseed )
     torch.manual_seed( self.rseed ) 
     torch.set_default_dtype(torch.float64)

     self.main( )
 
 # Auxiliary functions    
 
  def writeLogFile ( self, cwd, errmatrix, cputime = 0):
      if not self.preTrained:
          with open(os.path.join(cwd,'prnn.log'), 'a') as loc:
     #         loc.write('\nElapsedCPUtime = ' + str(cputime[0][0,0]) + '\n')
              loc.write('TotalNumberofParameters = ' + str(errmatrix[0][0,1]) + '\n')
              loc.write('AbsoluteError = ' + str(errmatrix[0][0,0]) + '\n')
      else:    
          with open(os.path.join(cwd, 'prnnoffline.log'), 'w') as loc:
#              loc.write('ElapsedCPUtime = ' + str(cputime[0][0,0]) + '\n')
              loc.write('TotalNumberofParameters = ' + str(errmatrix[0][0,1]) + '\n')
              loc.write('AbsoluteError = ' + str(errmatrix[0][0,0]) + '\n')

  def writeOutFile(self, cwd, output, cputime=0):
    if self.preTrained:
        with open(os.path.join(cwd, 'prnnoffline.out'), 'w') as loc:
            for i in range(len(output)):
                line = ' '.join(str(x) for x in output[i, :])
                loc.write(line[1:-1] + '\n')
                if (i + 1) % 60 == 0:
                    loc.write('\n')
      
  def writeErrFile ( self, cwd, errmatrix ):
      if not self.preTrained:
          with open(os.path.join(cwd, 'prnn.err'), 'w') as loc:
              loc.write(str(errmatrix[0][0,0]))      
      else:
          with open(os.path.join(cwd, 'prnnoffline.err'), 'w') as loc:
              loc.write(str(errmatrix[0][0,0]))      

  def calcError(self, combined, combinednndisp, nplot, nstep, ncomp = 3, lb = -1.0, ub = -1.0, ): 
    mse = np.array([])
    mserel = np.array([])
    for i in range ( nplot ): 
        init = i*nstep
        end = init + nstep
        if ( lb > -1.0 ):
           initb = init + lb
#           print('init ', init)
           endb = initb + (ub-lb)
        else:      
           initb = init
           endb = end 
        mse = np.append(mse, np.sqrt(((combined[initb:endb,3:6] - combinednndisp[initb:endb,3:6])**2).sum(axis=1))) 
        mserel = np.append(mserel, np.sqrt(((combined[initb:endb,3:6] - combinednndisp[initb:endb,3:6])**2).sum(axis=1))/np.sqrt((combined[initb:endb,3:6]**2).sum(axis=1)))
       # print(combined[initb:endb,3:6])
    return mse.mean(axis=0), mserel.mean(axis=0)

  def count_parameters(self, model):
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            print('Name ', name, ' values\n', torch.ravel(parameter.data))
            total_params+=param
      #  print(table)
        print(f"Total Trainable Params: {total_params}th")
        return total_params

  def readSmpFile(self, infilename):
    
        cwd = os.getcwd()
        infile = open(os.path.join(cwd, infilename), "r")
        tdata = infile.readlines()
        
        infile.close()
        sizetdata = len(tdata)
             
        for i in range(sizetdata):
            tdata[i] = tdata[i].split()
            
        tdata = [x for x in tdata if x != []]
           
        sizetdata = len(tdata)
        
        print("# of samples: ", sizetdata)
               
        y = np.array([np.array(xi) for xi in tdata])
        
        smpplt = y.copy()
        smpplt = np.array(smpplt, dtype=np.float64)
                       
        return smpplt
        
  def readData(self, training, timesteps):          
        # Getting data from sample files    
        
        for i in range(len(training)):
            if i == 0:
                trdataset = self.readSmpFile(training[i])
            else:
                trdataset = np.append(trdataset, self.readSmpFile(training[i]))
                trdataset = trdataset.reshape(int(len(trdataset)/6), 6)
        
        # Get number of load cases w/ RVE
        
        nlctr = int(len(trdataset)/timesteps)
        
        print('# load cases (total): ', nlctr)
        print('# length data: ', trdataset.shape)
           
        # Prepare data for training the NN
        
        trstrain = trdataset[:, 0:3]
        trsig = trdataset[:, 3:6]
        
        return trstrain, trsig
    
  def normalize_2d (self, data):
    normdata = np.empty_like(data)
    for i in range ( data.shape[1] ):
      normdata[:,i] = 2.0*((data[:,i] - np.min(data[:,i])) / (np.max(data[:,i]) - np.min(data[:,i])))-1.0
    return normdata

  def writeNmlFile ( self, cwd, filename, data ):
      with open (os.path.join(cwd, filename), 'w') as loc:
         for i in range ( data.shape[1] ):
             loc.write(str(np.max(data[:,i])) + ' ' + str(np.min(data[:,i])) + '\n')
      
  # Normalization between -1 and 1      
      
  def applyNorm ( self, cwd, filename, data ):
    infile = open(os.path.join(cwd, filename), "r")
    bounds = infile.readlines()
    infile.close()
    for i in range(len(bounds)):
        bounds[i] = bounds[i].split()
    bounds = np.array(bounds, dtype=np.float64)
    normdata = np.empty_like(data)
    for i in range ( data.shape[1] ):
      normdata[:,i] = 2.0*((data[:,i] - bounds[i, 1]) / (bounds[i, 0] - bounds[i, 1]))-1.0
    return normdata

  def main ( self ):
        
    #Pre-processing   
    
    maxIt = self.maxIt
    maxStallIt = self.earlyStop 
    n_features = 3
   
    trrve = self.trainingDataset
    print('Reading training (or testing) data from ', trrve)
    nTimeSteps = int(trrve[0][:-5][-3:])       
    trstraintemp, trsigtemp = self.readData(trrve, nTimeSteps)
           
    sequence_length = nTimeSteps
    ntr = len(trrve)
    
    # Structuring the data 
        
    layersizelist = np.array([self.ls])  
    nlctrlist = np.array([self.trsize])  # Number of curves considered for training
    
    # Store error for evaluation
    
    errmatrix = np.array([])
    cputime = np.array([])
    
    # Running a loop with different layer sizes    
    # and different training set sizes
    
    for ls in layersizelist:  
        for ncurves in nlctrlist:     
            nlctr = ncurves           
            n_data = int(trsigtemp.shape[0]/(ntr*sequence_length)) 
           
            if not self.preTrained:                    
                if (nlctr <= n_data):
                    print("\nAttention: Using only ", nlctr, " out of ", n_data," curves to train the PRNN.\n")
                    if ( nlctr + self.skipFirst > n_data ):
                        print('\nAttention: Insufficient number of curves '
                              'available for the specified size of validation set. '
                              'Reducing it to ', n_data-nlctr, ' curves only.')
                        self.skipFirst = n_data-nlctr
                    rangeofidx = range(0, n_data-self.skipFirst)
                    #shuffled = np.asarray(random.sample(rangeofidx, self.trsize))
                    shuffled = np.random.choice(rangeofidx, size=self.trsize, replace=False)
                    #shuffled = random.randint(n_data-self.skipFirst, size =  self.trsize)
                    samples = shuffled + self.skipFirst
                    valsamples = np.arange(0, self.skipFirst)
                    
                    # Selecting validation set
                    
                    for i in range(self.skipFirst):
                        initval = valsamples[i]*sequence_length
                        endval = initval + sequence_length
                        if i == 0:
                            valstrain = trstraintemp[initval:endval, :]
                            valsig = trsigtemp[initval:endval, :]
                        else:
                            valstrain = np.vstack([valstrain, trstraintemp[initval:endval, :]])
                            valsig = np.vstack([valsig, trsigtemp[initval:endval, :]])

                    # Selecting training set                   

                    for i in range(self.trsize):
                        inittr = samples[i]*sequence_length
                        endtr = inittr + sequence_length
                        if i == 0:
                            trstrain = trstraintemp[inittr:endtr, :]
                            trsig = trsigtemp[inittr:endtr, :]
                        else:
                            trstrain = np.vstack([trstrain, trstraintemp[inittr:endtr, :]])
                            trsig = np.vstack([trsig, trsigtemp[inittr:endtr, :]])
                    n_data = nlctr
                    print('Idx of samples used for validation: ', valsamples)                            
                    print('Idx of samples used for training: ', samples,'\n')

                else:
                    print("\nAttention: Number of curves available for training is lower than "
                          "expected. Using all ", n_data, " to train the PRNN. "
                          "This means that NO validation set will be considered.\n")
                    trstrain = trstraintemp
                    trsig = trsigtemp 
                    self.skipFirst = 0
                    
                cwd = os.getcwd( )        
                prnnfilename = os.path.join (cwd, 'prnn' + str(self.loadtype) + '_' + str(self.rseed) + '_' + str(ls) + '_' + str(n_data) + 'curves1layer.pth')
            else:
                if ( self.evalncurves <= n_data ):
                    print("\nEvaluation mode: Number of curves being evaluated ", self.evalncurves, 
                          "out of ", n_data, ". \n")
                    n_data = self.evalncurves
                    inittr = 0 
                    endtr = n_data*sequence_length
                    inittrr =  int(trsigtemp.shape[0]/ntr) 
                    endtrr  =  inittrr + endtr
                    trstrain = np.vstack([trstraintemp[inittr:endtr, :], trstraintemp[inittrr:endtrr, :]])
                    trsig = np.vstack([trsigtemp[inittr:endtr, :], trsigtemp[inittrr:endtrr, :]])
                else:
                    print("\nAttention: Number of curves available is lower than "
                          "expected. Using all ", n_data, " to evaluate the BNN.\n")
                    trstrain = trstraintemp
                    trsig = trsigtemp

                cwd = os.getcwd( )        
                prnnfilename = os.path.join (cwd, 'prnn' + str(self.loadtype) + '_' + str(self.rseed) + '_' + str(ls) + '_' + str(ncurves) + 'curves1layer.pth')
                print('Reading parameters (weights+biases) from ', prnnfilename)
                    
            n_data = ntr*n_data
            
            if not self.preTrained:
       #        print('Before normalize ', trstrain)
                normalizedtrstrain = self.normalize_2d(trstrain)
                cwd = os.getcwd()
                self.writeNmlFile (cwd, 'prnn.nml', trstrain)
                if self.normalize: trstrain = normalizedtrstrain
            else:
                cwd = os.getcwd()
                if self.normalize: 
                    normalizedtrstrain = self.applyNorm (cwd, 'prnn.nml', trstrain )
                    trstrain = normalizedtrstrain
                        
            x_train = trstrain.reshape([n_data,sequence_length, n_features])  
            y_train = trsig.reshape([n_data, sequence_length, n_features])
                    
            # Passing data to dataloader
            
            dataset = timeseries(x_train, y_train) 
            train_loader = DataLoader(dataset, shuffle=True, batch_size= self.batchSize)
     
            # Init main loop when training 
            
            if not self.preTrained:
     
              x_val = valstrain.reshape([self.skipFirst,sequence_length, n_features])  
              y_val = valsig.reshape([self.skipFirst, sequence_length, n_features])

              valdataset = timeseries(x_val, y_val) 
              val_loader = DataLoader ( valdataset, shuffle = False )         
     
              epochs = maxIt
              modeltr = neural_network(n_features=3,output_length=3, 
                                       bulk = self.bulkPts, cohesive = self.cohPts,
                                       dev = device).to(device)
               
              criterion = torch.nn.MSELoss()
               #optimizer = torch.optim.Adam(model.parameters(),lr=1.0e-5, weight_decay=1e-5)
              optimizer = torch.optim.Adam(modeltr.parameters(),lr=1.0e-2)                   
              
              if self.warmStart:              
                 modeltr.load_state_dict(torch.load(prnnfilename))
                 modeltr.train()  
                 print('\nWARNING: Warm start will resume training based on '
                       'parameters stored in ', prnnfilename)
                            
              # Starting training loop 
              
              stallIt = 0
              torch.autograd.set_detect_anomaly(True)
              os.system('rm rnn.log')
              cwd = os.getcwd()
              
              with open (os.path.join(cwd, 'rnn.log'), 'w' ) as loc:
                loc.write('Index of training directions ' + str(samples) + '\n')  
                for i in range(epochs):
                  running_loss = 0
                  for j, data in enumerate(train_loader):
                      y_pred = modeltr( data[:][0] )
                      loss = criterion(y_pred, data[:][1])
                      optimizer.zero_grad()  # Clears existing gradients
                      loss.backward()
                      optimizer.step()
                      running_loss += loss.item()

                  if i%self.writeEvery == 0:
                      
                      # Calculate loss on the validation set every N epochs
                      
                      if ( self.skipFirst > 0 ):
                          with torch.no_grad():
                            modeltr.eval()
                            running_loss_val = 0
                            for j, data in enumerate(val_loader):
                                y_predval = modeltr ( data[:][0] ) 
                                loss = criterion ( y_predval, data[:][1])
                                running_loss_val += loss.item()
                            running_loss_val /= self.skipFirst
                            modeltr.train()
                      else:
                            running_loss_val = running_loss
                            
                      print('Epoch ', i, ' training loss ', running_loss)
                      loc.write('Epoch ' + str(i) + ' (training) loss = ' + str(running_loss))
                      print('Epoch ', i, ' validation loss ', running_loss_val )
                      loc.write('\nEpoch ' + str(i) + ' (validation) loss = ' + str(running_loss_val))

                      if ( i == 0 ): 
                          prev = running_loss_val
                          
                     # Only update file with best parameters if validation loss
                     # is smaller than the historical best so far
                              
                      if ( running_loss_val <= prev ):
                          prev = running_loss_val
                          best_model_state = copy.deepcopy(modeltr.state_dict())
                          torch.save(best_model_state, prnnfilename)
                          print('Saved model in ', prnnfilename)
                          stallIt = 0
                      else:
                          # Keeping track of how many epochs have gone 
                          # without any improvement on validation set
                          
                          stallIt += self.writeEvery
                          
                      loc.write('\nNumber of stall epochs: ' + str(stallIt) + '\n')
                      if ( stallIt == maxStallIt ):
                         loc.write('Max Number of stall epochs reached!\n')
                         break
              
                
            # Once training loop is finished, evaluate NN with the optimal
            # set of weights obtained previously (or when running evaluation
            # mode)
            
            model = neural_network(n_features=3,output_length=3, 
                                   bulk = self.bulkPts, cohesive = self.cohPts,
                                   dev = device).to(device)
            model.load_state_dict(torch.load(prnnfilename))
            model.eval()
          
            if self.evalError:            

                print('\nPredicting on validation/test set\n')
                
                if not self.preTrained:
                    # Same as validation set, but with no shuffle and batch = 1
                    valdataset = timeseries( x_val, y_val) #dataloader
                    test_loader = DataLoader ( valdataset, shuffle=False,batch_size=1 )    
                    stiffmat = torch.zeros((x_val.shape[0], sequence_length, 3, 3))
                else:
                    # Same as training/testing set, but with no shuffle and batch = 1
                    test_loader = DataLoader ( dataset, shuffle = False, batch_size = 1)
                    stiffmat = torch.zeros((x_train.shape[0], sequence_length, 3, 3))
                                        
                for j, data in enumerate(test_loader):
                     print('Progress ', j+1, '/', len(test_loader))
                     if ( self.printTangent ):
                        strain_in = data[:][0].clone().requires_grad_(True)
                        stiff = torch.autograd.functional.jacobian(model, strain_in ) 
                        print('Tangent stiffness shape (autodiff): ', stiff.shape)
                        for tstep in range ( sequence_length ):
                           stiffmat[j, tstep, 0, :] = stiff[0][tstep][0][0][tstep]
                           stiffmat[j, tstep, 1, :] = stiff[0][tstep][1][0][tstep]
                           stiffmat[j, tstep, 2, :] = stiff[0][tstep][2][0][tstep]
                           print('Stiff time step ', tstep, ' batch ', j, ':\n',  stiffmat[j, tstep, :, :])
                     test_pred = model(data[:][0] ).cpu()
                     if ( j == 0 ):
                         test_pred_array = np.array(test_pred.detach().numpy().reshape(-1))
                     else:
                         test_pred_array = np.append(test_pred_array, test_pred.detach().numpy().reshape(-1))
                   
                if not self.preTrained:
                    combined = np.column_stack ( ( valstrain, valsig ) )
                    combinedbnn = np.column_stack ((valstrain, test_pred_array.reshape([y_val.shape[0]*y_val.shape[1], 3])))
                else:
                    combined = np.column_stack ( ( trstrain, trsig ) )
                    combinedbnn = np.column_stack ((trstrain, test_pred_array.reshape([y_train.shape[0]*y_train.shape[1], 3])))

                #print(combinedbnn)
                # Detail number, name and value of optimal parameters
 
                print ('\nOptimal parameters')
    
                trparam = self.count_parameters(model)
            
                # Calculate error 
            
                errabs, errrel = self.calcError(combined, combinedbnn, test_pred_array.shape[0], sequence_length, n_features)
                errmatrix = np.append(errmatrix, [errabs, trparam])
                
                print('\nError ', errabs)
              
                errmatrix = np.reshape(errmatrix, [len(layersizelist), len(nlctrlist), 2])   
                #cputime = np.reshape(cputime, [ len(layersizelist), len(nlctrlist), 1])
               
               # Write log and error files 
               
                cwd = os.getcwd() 
                self.writeLogFile(cwd, errmatrix)
                self.writeErrFile(cwd, errmatrix)

                roundTo = 8
                testa = np.round(test_pred_array,roundTo)
                outdata = np.column_stack ((trstrain, testa.reshape([y_train.shape[0]*y_train.shape[1], 3])))
                self.writeOutFile(cwd, outdata)
                
           #     print("========FINAL ========= Model's state_dict:")
           #     for name, param_tensor in model.state_dict().items():
           #       print(name, "\n", param_tensor.size())
           #       print(name, "\n", param_tensor)
