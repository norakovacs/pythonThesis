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
from customlayers import softLayer, blockLayer, blockDecLayer, symmLayer, leakyr, softplusb, mid
import sys
import numpy as np
import time
import os

# Architecture no.6
# Damage input to J2, softplus activation function on strain-displacement jump relation WITH BIAS

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
        self.in_bulk = self.in_size + self.cohPts
                
        print('Input size ', self.in_size)
        print('Material layer size ', self.hidden_size)
        print('Output layer size ', self.output_length)
        
        # Defining layers 
        
        # Linear -> Regular dense layer
        # softLayer -> Regular dense layer with sotfplus on weights (customized)
        
#        self.fc1 = blockLayer(in_features=self.in_size,out_features=self.hidden_size, device = device, bias = False)
        #self.fc1 = nn.Linear(in_features=self.in_size,out_features=self.hidden_size, device = self.device, bias = False)
      #   self.fc11 = lin0(in_features=self.in_size,out_features=self.coh_size, device = self.device, bias = False)
      #   self.fc112 = leakyr(in_features=self.in_size,out_features=self.coh_size, device = self.device, bias = True)
        self.fc11 = leakyr(in_features=self.in_size,out_features=self.coh_size, device = self.device, bias = True)
      #   self.mid = mid(in_featuresone=self.in_size, in_featurestwo=self.cohPts, out_features=self.in_bulk,device=self.device)
        self.mid = mid()
        self.fc12 = nn.Linear(in_features=self.in_bulk,out_features=self.bulk_size, device = self.device, bias = False)
        self.fc2 = softLayer(in_features=self.bulk_size,out_features=self.output_length, device = self.device, bias = False)

    def getOutputMatPts(self,x):
        # Equivalent to propagate 
        
        batch_size, seq_len, _ = x.size()
        
        output =  x.clone()
        # print(self.in_bulk)
        # sys.exit()

        out = torch.zeros([batch_size,seq_len, self.output_length]).to(self.device)
        mid = torch.zeros([batch_size,seq_len, self.in_bulk]).to(self.device)

        # Create material models
        
        childb = J2Tensor.J2Material() 
        childc = TuronCohesiveMat.TuronCohesiveMat()
        
        # Create and configure integration points

        localstrains = torch.zeros([batch_size, seq_len, self.ls]) 
        localstresses = torch.zeros([batch_size, seq_len, self.ls]) 
        localhistory = torch.zeros([batch_size, seq_len, self.bulkPts*1+self.cohPts*1]) 
        
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
               # x = self.fc11(output[j,t,:])
               # outputt1 = self.fc112(x)
               outputt1 = self.fc11(output[j,t,:])
               localstrains[j, t, self.bulkPts*3:] = outputt1
               #print('Local strain at time step ', t, ' from curve ', j, ': ', outputt1)
               #sys.exit()

               # Evaluating cohesive models

               for ip in range ( self.cohPts ):
                   #initipc = self.bulkPts*3 + ip*2
                   #endipc = self.bulkPts*3 + (ip+1)*2
                #    self.timer.start()
                   outputt1[ip*2:(ip+1)*2], mid[j,t,self.in_size+ip] = childc.update(outputt1[ip*2:(ip+1)*2], j*self.cohPts + ip)
                #    print("update function in Turon model done in ", self.timer.stop(), ' seconds')
                #    print(childc.preHist_[ip].damage)
                #    print(childc.newHist_[ip].damage)
                   childc.commit(j*self.cohPts + ip)
                   d = childc.getHistory(ip)
                   #print(d)
                   localhistory[j,t, self.bulkPts+ip:self.bulkPts*1+(ip+1)*1] = d
                   
               #print(dam)
               localstresses[j, t, self.bulkPts*3:]  =  outputt1

               mid[j,t,:self.in_size] = output[j,t,:]
               outputt2 = self.fc12(mid[j,t,:])
               localstrains[j, t, :self.bulkPts*3] = outputt2

               # Evaluating bulk models
                  
               for ip in range ( self.bulkPts ):
                #    self.timer.start()
                   outputt2[ip*3:(ip+1)*3] = childb.update(outputt2[ip*3:(ip+1)*3], j*self.bulkPts + ip)
                #    print("update function in J2 model done in ", self.timer.stop(), ' seconds')
                   childb.commit(j*self.bulkPts + ip)
                   localhistory[j,t, ip:(ip+1)] = childb.getHistory(ip)[1]
    
              # Decoder ( homogenization )
               #print('Local stress at time step ', t, ' from curve ', j, ': ', outputt2)
               localstresses[j, t, :self.bulkPts*3]  =  outputt2            
               outputt2 = self.fc2(outputt2)
               #print('Homogenized stress at time step ', t, ' from curve ', j, ': ', outputt2)
               out[j,t, :] = outputt2.view(-1,self.output_length)
               #sys.exit()
               

        output=out.to(self.device)
        return output, localstrains, localstresses, localhistory


    def forward(self,x):
        # Equivalent to propagate 
        
        batch_size, seq_len, _ = x.size()
        
        output =  x.clone()
        c = x.clone()
        # print(self.in_bulk)
        # sys.exit()

        out = torch.zeros([batch_size,seq_len, self.output_length]).to(self.device)
        #mid = torch.zeros([batch_size,seq_len, self.in_bulk]).to(self.device)
        #damage = torch.zeros(self.cohPts).to(self.device)

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
               # b = self.fc11(output[j,t,:])
               # outputt1 = self.fc112(b)
               outputt1 = self.fc11(output[j,t,:])
               #print(outputt1)

               #print('Local strain at time step ', t, ' from curve ', j, ': ', outputt1)
               #sys.exit()
               cwd = os.getcwd()


               # Evaluating cohesive models
               #print(outputt1)
               for ip in range ( self.cohPts ):
                   #initipc = self.bulkPts*3 + ip*2
                   #endipc = self.bulkPts*3 + (ip+1)*2
                #    self.timer.start()
                   d= childc.update(outputt1[ip*2:(ip+1)*2], j*self.cohPts + ip)
                   #midini = torch.cat((output[j,t,:],d))
                #    print("update function in Turon model done in ", self.timer.stop(), ' seconds')
                #    print(childc.preHist_[ip].damage)
                #    print(childc.newHist_[ip].damage)
                   childc.commit(j*self.cohPts + ip)
                  #  d = childc.getHistory(ip)
                  #  print(mid[j,t,self.in_size+ip])
                   if ip ==0:
                       midini = torch.cat((output[j,t,:],d))
                   else:
                       midini = torch.cat((midini,d))
                   #damage[ip] = d
                   print('damage is ',d)
                   if t ==0 or t==60:
                       with open (os.path.join(cwd, 'damage.log'), 'a+' ) as l:
                          l.write('timestep ' + str(t)+', damage is ' + str(d) +'\n')
                          l.close()
                   if t ==0 and d>0.99:
                     raise Exception('damage starts from 1!')
               #midini = torch.cat((output[j,t,:],damage))
                   
               #mi = self.mid(output[j,t,:],damage)
               #print(mi)
               #print(midini)
               outputt2 = self.fc12(midini)
               #print(outputt2)

               # Evaluating bulk models
                  
               for ip in range ( self.bulkPts ):
                #    self.timer.start()
                   outputt2[ip*3:(ip+1)*3] = childb.update(outputt2[ip*3:(ip+1)*3], j*self.bulkPts + ip)
                #    print("update function in J2 model done in ", self.timer.stop(), ' seconds')
                   childb.commit(j*self.bulkPts + ip)
    
              # Decoder ( homogenization )
               #print('Local stress at time step ', t, ' from curve ', j, ': ', outputt2)
                           
               outputt2 = self.fc2(outputt2)
               #print('Homogenized stress at time step ', t, ' from curve ', j, ': ', outputt2)
               out[j,t, :] = outputt2.view(-1,self.output_length)
               #sys.exit()
               

        output=out.to(self.device)
        return output
