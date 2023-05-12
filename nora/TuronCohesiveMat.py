#!/usr/bin/env python
"""

Turon material model

"""

import torch
import numpy as np
print(f"torch version: {torch.__version__}")
import math 
import sys
# import torchviz
# from torchviz import make_dot

torch.set_default_dtype(torch.float64)
# torch.set_default_dtype(torch.float32)

deepcopy = True
import warnings
warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)

class TuronCohesiveMat:
    def __init__(self):
        self.dummy_ = 5e7

        self.f2t_ = 60.0
        self.f6_  = 60.0
        self.gI_  = 0.874
        self.gII_ = 1.717
        self.eta_ = 1.0

        self.preHist_ = []
        self.latestHist_ = []
        self.newHist_ = []

        self.deltaNF = 2 *  self.gI_ / self.f2t_
        self.deltaSF = 2 * self.gII_ /  self.f6_
        self.deltaN0 = self.f2t_ / self.dummy_
        self.deltaS0 = self.f6_ / self.dummy_

        self.deltaN02_ = self.deltaN0**2
        self.deltaS02_ = self.deltaS0**2
        self.deltaN0F_ = self.deltaN0 * self.deltaNF
        self.deltaS0F_ = self.deltaS0 * self.deltaSF

        self.f2t2_ = self.f2t_**2
        self.f62_  = self.f6_**2

        self.omega_max = 1 - 1e-12
        self.rank_ = 2
        self.EPS = 1e-10
    
    def getHistory(self, ip):
        return self.latestHist_[ip].HistVec()

    def setHistory(self, damage, ip):
        self.preHist_[ip].damage = damage.clone()
        self.latestHist_[ip].damage = damage.clone()
        self.newHist_[ip].damage = damage.clone()

    def configure(self, npoints):
        for i in range(npoints):
            self.preHist_.append( self.Hist() )
            self.newHist_.append( self.Hist() )
            self.latestHist_.append( self.Hist() )

    def update(self, jump, ip):
        jump_new = jump.clone()

        if jump_new[0] < 0:
            jump_new[0] = 0.0
        #print('jumpcorr is ', jump_new)

        delta = torch.linalg.norm(jump_new)
        deltaS = torch.linalg.norm(jump_new[1:])

        if deltaS + jump_new[0] > 0:
            beta = deltaS / (deltaS + jump_new[0])
        else:
            beta = torch.tensor(0.0, dtype=torch.float64).requires_grad_(True)

        #eqn 42
        B = beta**2.0 / (1.0 + 2.0*beta**2.0 - 2.0*beta)

        #eqn 45
        delta0 = torch.sqrt(self.deltaN02_ + ( self.deltaS02_ - self.deltaN02_ ) * B**self.eta_)

        #eqn 36
        deltaF = (self.deltaN0F_ + ( self.deltaS0F_ - self.deltaN0F_ ) * B**self.eta_) / delta0

        #calculating damage
        hisDam = ( delta - delta0 ) / ( deltaF - delta0 )

        #print('damage previous is ',self.latestHist_[ip].damage)
        if hisDam - self.preHist_[ip].damage > - 1.e-14:
            loading = 1
        else:
            loading = 0
            hisDam = self.preHist_[ip].damage

        if hisDam > 0:
            if loading:
                damage = hisDam * deltaF / delta #eqn 31
            else:
                damage  = hisDam * deltaF / (delta0 + hisDam * ( deltaF - delta0 ))
        else:
            hisDam = damage =  0
        
        if hisDam >= self.omega_max :
            hisDam = damage = self.omega_max
            loading = 0

        #calculating secant stiffness and traction
        stiff = torch.zeros((2, 2), dtype=torch.float64)
        
        if jump_new[0] < 0:
            stiff[0, 0] = self.dummy_
        else:
            stiff[0, 0] = (1.0 - damage) * self.dummy_

        for i in range(1, self.rank_):
            stiff[i, i] = (1.0 - damage) * self.dummy_
        traction = torch.matmul(stiff, jump_new)

        #print('stiffness matrix from update function: ', stiff)

        self.newHist_[ip].damage = hisDam
        self.latestHist_[ip] = self.newHist_[ip]

        # print('Backwards testing.')
        # hisDam.backward()
        # print('Gradient w.r.t. to jump is ', jump.grad)
        # sys.exit()

        return traction

    def elasticUpdate(self, traction, stiff, jump):
        for i in range(self.rank_):
            traction[i] = self.dummy_ * jump[i]
            for j in range(self.rank_):
                stiff[i, j] = self.dummy_ if i == j else 0.0
    
    def commit(self,ipoint):
        self.preHist_[ipoint].damage = torch.tensor(self.newHist_[ipoint].damage).clone()
        self.latestHist_[ipoint].damage = torch.tensor(self.preHist_[ipoint].damage).clone()
    
    class Hist:
        def __init__(self):
            self.damage = torch.zeros(1)

        def HistVec(self):
            return self.damage

