#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 15:44:35 2023

@author: malvesmaia
"""
from torch import nn
import torch
import J2Tensor 
import TuronCohesiveMat
from customlayers import softLayer, blockLayer, blockDecLayer, symmLayer
import sys
import numpy as np

class neural_network(nn.Module):
    def __init__(self,n_features,output_length, bulk, cohesive, dev):
        super(neural_network,self).__init__()

        self.device = dev
        self.bulkPts = bulk
        self.cohPts = cohesive
        self.nIntPts = self.bulkPts + self.cohPts
        self.ls = self.bulkPts*3 + self.cohPts*2
        self.hidden_size = self.ls
        self.coh_size = self.cohPts*2
        self.bulk_size = self.bulkPts*3
        self.n_layers = 1        
        self.in_size = n_features
        self.output_length = output_length
        self.in_bulk = self.in_size + self.coh_size
                
        print('Input size ', self.in_size)
        print('Material layer size ', self.hidden_size)
        print('Output layer size ', self.output_length)
        
        # Defining layers 
        
        # Linear -> Regular dense layer
        # softLayer -> Regular dense layer with sotfplus on weights (customized)
        
#        self.fc1 = blockLayer(in_features=self.in_size,out_features=self.hidden_size, device = device, bias = False)
        #self.fc1 = nn.Linear(in_features=self.in_size,out_features=self.hidden_size, device = self.device, bias = False)
        self.fc11 = nn.Linear(in_features=self.in_size,out_features=self.coh_size, device = self.device, bias = False)
        self.fc12 = nn.Linear(in_features=self.in_size,out_features=self.bulk_size, device = self.device, bias = False)
        self.fc2 = softLayer(in_features=self.bulk_size,out_features=self.output_length, device = self.device, bias = False)
        
    def forward(self,x):
        
        # Equivalent to propagate 
        
        batch_size, seq_len, _ = x.size()
        
        output =  x.clone()
        # print(self.in_bulk)
        # sys.exit()

        out = torch.zeros([batch_size,seq_len, self.output_length]).to(self.device)
        #mid = torch.zeros([batch_size,seq_len, self.in_bulk]).to(self.device)

        # Create material models
        
        childb = J2Tensor.J2Material() 
        childc = TuronCohesiveMat.TuronCohesiveMat()
        
        # Create and configure integration points
        
        ip_pointsb = batch_size*self.bulkPts
        ip_pointsc = batch_size*self.cohPts
        childb.configure( ip_pointsb )        
        childc.configure( ip_pointsc )   
        
      #  print('Created ', ip_pointsb, ' bulk material points')
      #  print('Created ', ip_pointsc, ' cohesive material points')

      #  print('Batch size ', batch_size, ' Sequence length ', seq_len)

        # Process each curve at a time

        for j in range ( batch_size ):            
            for t in range(seq_len):
                
               # Encoder ( dehomogenization )
            #   print('Macro strain at time step ', t, ' from curve ', j, ': ', output[j,t,:])
               outputt1 = self.fc11(output[j,t,:])
               #print('Local strain at time step ', t, ' from curve ', j, ': ', outputt1)
               #sys.exit()

               # Evaluating cohesive models
               dam = np.zeros((self.cohPts, seq_len))
               load = np.zeros((self.cohPts, seq_len))

               for ip in range ( self.cohPts ):
                   #initipc = self.bulkPts*3 + ip*2
                   #endipc = self.bulkPts*3 + (ip+1)*2
                   outputt1[ip*2:(ip+1)*2], loading = childc.update(outputt1[ip*2:(ip+1)*2], j*self.cohPts + ip)
                #    print(childc.preHist_[ip].damage)
                #    print(childc.newHist_[ip].damage)
                   childc.commit(j*self.cohPts + ip)
                   d = childc.getHistory(ip)
                   dam[ip,t] = d
                   load[ip,t] = loading
                   #print(loading)
                   
               #print(dam)
               
               outputt2 = self.fc12(output[j,t,:])
               #print('input to j2: ', outputt2)

               # Evaluating bulk models
                  
               for ip in range ( self.bulkPts ):
                   outputt2[ip*3:(ip+1)*3] = childb.update(outputt2[ip*3:(ip+1)*3], j*self.bulkPts + ip)

                   if load[ip,t]==0:
                    outputt2[ip*3:(ip+1)*3] = outputt2[ip*3:(ip+1)*3] * (1 - dam[ip,t])

                   childb.commit(j*self.bulkPts + ip)
    
              # Decoder ( homogenization )
               #print('Local stress at time step ', t, ' from curve ', j, ': ', outputt2)
                           
               outputt2 = self.fc2(outputt2)
               #print('Homogenized stress at time step ', t, ' from curve ', j, ': ', outputt2)
               out[j,t, :] = outputt2.view(-1,self.output_length)
               #sys.exit()
               

        output=out.to(self.device)
        return output