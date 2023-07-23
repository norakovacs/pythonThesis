#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 15:34:27 2023

@author: malvesmaia
"""
from torch import nn
import torch

# ------------------- Definition of customized layers ------------------------
    
class softLayer(nn.Module): 
    from torch import Tensor
    
    # This layer applies the softplus function to its weights
    # so that all weights associated to it are stictly positive
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(softLayer, self).__init__()
    
        self.in_features = in_features
        self.out_features = out_features
        self.sp = nn.Softplus()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        import math 
    
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
    #   print('input for last layer: ', input)
    #   print('weight in last layer: ', self.weight)
      return nn.functional.linear(input, self.sp(self.weight), self.bias) 
  
class shiftedSoftplus(nn.Module): 
    from torch import Tensor
    
    # This layer applies a shifted softplus function to convert macroscopic
    # strains to displacement jumps. Shift is set to 5.
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(shiftedSoftplus, self).__init__()
    
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        import math 
    
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
      return nn.functional.softplus(nn.functional.linear(input, self.weight, self.bias) -5.0) 
    
class softplusb(nn.Module): 
    from torch import Tensor
    
    # This layer applies a softplus function with bias to convert macroscopic
    # strains to displacement jumps.
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(softplusb, self).__init__()
    
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        import math 
    
        bound = 0.01
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 0 #1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            #print(self.bias)

    def forward(self, input: Tensor) -> Tensor:
      #print(self.bias)
      return nn.functional.softplus(nn.functional.linear(input, self.weight, self.bias)) 

class leakyrel(nn.Module): 
    from torch import Tensor
    
    # This layer applies a leaky relu function with bias to convert macroscopic
    # strains to displacement jumps.
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(leakyrel, self).__init__()
    
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        import math 
    
        bound = 0.01
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 0 #1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            #print(self.bias)

    def forward(self, input: Tensor) -> Tensor:
      print(self.bias)
      return nn.functional.leaky_relu(nn.functional.linear(input, self.weight, self.bias)) 

class tanh(nn.Module): 
    from torch import Tensor
    
    # This layer applies a tanh function in the hidden layer.
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(tanh, self).__init__()
    
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        import math 
    
        #nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.uniform_(self.weight, -0.0005, 0.0005)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
      return nn.functional.tanh(nn.functional.linear(input, self.weight, self.bias)) 

class leakyr(nn.Module): 
    from torch import Tensor
    
    # This layer applies a leaky reku function with bias to convert macroscopic
    # strains to displacement jumps.
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(leakyr, self).__init__()
    
        self.in_features = in_features
        self.out_features = out_features
        self.sp = nn.Softplus()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weightleaky = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        self.biasw = torch.zeros(out_features, **factory_kwargs)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self) -> None:
        import math 
    
        #bound = 0.01
        #nn.init.uniform_(self.weight, -bound, bound)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        # bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.weightleaky, -9., -8.)

        if self.bias is not None:
            bound = 0.01 #1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, math.log(math.exp(0.001)-1.), math.log(math.exp(0.1)-1.))
            #print(self.bias)
    
    def leaky_mod(self, x,bias, w) -> Tensor:
        xr = x.clone()
        print('bias: ',bias)
        if x[0] >= bias[0]:
            xr[0] = w[0] * (x[0] - bias[0])
        else:
            xr[0] = 0.01 * w[0] * (x[0] - bias[0])
        for i in range(1,x.size(0)):
            if x[i] >= 0.:
                xr[i] = w[i] * (x[i])
            else:
                xr[i] = 1e-5 * w[i] * (x[i])
            #xr[i] = res
        #print(xr)
        return xr#.clone().requires_grad_(True)

    def forward(self, input: Tensor) -> Tensor:
      x = nn.functional.linear(input, self.weight, self.biasw)
      #print('x: ',x)
    #   print('weight: ',self.weight)
    #   print('weightl: ',self.weightleaky)
    #   print(nn.Softplus(self.bias))
      return self.leaky_mod(x,self.sp(self.bias),self.sp(self.weightleaky)) 

class mid(nn.Module): 
    from torch import Tensor
    
    def __init__(self,
                 device=None, dtype=None) -> None:
        
        super(mid, self).__init__()
    

    def forward(self, inputone: Tensor, inputtwo:Tensor) -> Tensor:
      return torch.cat((inputone,inputtwo))

class blockLayer(nn.Module): 
    from torch import Tensor
    
    # This layer multiplies the activations from the previous layer in groups
    # of size of the input layer and the weight matrix of each subgroup is
    # originated from a LLt decomposition (i.e. always SPD)
    
    # Each block of weights is composed of LLt, from which follows:
    # Lt = [[softplus(weight_0) weight_1           weight_2]
    #      [0.                  softplus(weight_3) weight_4]    
    #      [0.                  0.                 softplus(weight_5)]]
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(blockLayer, self).__init__()
    
        self.in_features = in_features
        self.nIntPts = int(out_features/3)
        self.out_features = out_features
        self.sp = nn.Softplus()
        self.weight = nn.Parameter(torch.empty((self.nIntPts, 6), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import math 
    
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, input: Tensor) -> Tensor:
      #print('Weights ', self.weight)
        
      if ( len(input.shape) == 1 ):
          batch_size = 1
          output = torch.zeros((self.nIntPts*3))
      else:
          batch_size = input.shape[0]
          output = torch.zeros((batch_size, self.nIntPts*3))
          
      for i in range ( self.nIntPts ):
          wMatrix = torch.zeros((3, 3))

          wMatrix[0, 0] = self.sp(self.weight[i, 0].clone())
          wMatrix[1, 1] = self.sp(self.weight[i, 3].clone())          
          wMatrix[2, 2] = self.sp(self.weight[i, 5].clone()) 
          wMatrix[0, 1] = self.weight[i, 1].clone()
          wMatrix[0, 2] = self.weight[i, 2].clone()
          wMatrix[1, 2] =  self.weight[i, 4].clone()
          
         # print('wmatrix ', wMatrix)
          
          wBlock = (torch.matmul(wMatrix.t(), wMatrix)).t()
          
          if ( batch_size == 1 ):
            output[i*3:(i+1)*3] = nn.functional.linear(input, wBlock, self.bias)
          else:         
            output[:, i*3:(i+1)*3] = nn.functional.linear(input, wBlock, self.bias)
      
      return output
  
class symmLayer(nn.Module): 
    from torch import Tensor
    
    # This layer should be tied to another layer, so that the same weights from
    # the original layer are are also used here (but transposed)
    
    def __init__(self, tied_to: nn.Linear, out_features: int,
                 bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(symmLayer, self).__init__()
    
        self.tied_to = tied_to
        self.nIntPts = self.tied_to.nIntPts
        self.out_features = out_features
        self.sp = nn.Softplus()
        if bias:
            self.bias = nn.Parameter(torch.Tensor(tied_to.in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        print('Warning: already initiated.')

    def forward(self, input: torch.Tensor) -> torch.Tensor:
    #  print('Weights from blockLayer ', self.tied_to.weight)
      if ( len(input.shape) == 1 ):
          batch_size = 1
      else:
          batch_size = input.shape[0]
              
      if batch_size == 1:
          output = torch.zeros((3))
      else:
          output = torch.zeros((batch_size, 3))
          
      for i in range ( self.nIntPts ):
          wMatrix = torch.zeros((3, 3))

          wMatrix[0, 0] = self.sp(self.tied_to.weight[i, 0].clone())
          wMatrix[1, 1] = self.sp(self.tied_to.weight[i, 3].clone())          
          wMatrix[2, 2] = self.sp(self.tied_to.weight[i, 5].clone()) 
          wMatrix[0, 1] = self.tied_to.weight[i, 1].clone()
          wMatrix[0, 2] = self.tied_to.weight[i, 2].clone()
          wMatrix[1, 2] =  self.tied_to.weight[i, 4].clone()
          
          wBlock = torch.matmul(wMatrix.t(), wMatrix)
          
        #  print('wmatrix copy', wMatrix)
          
          if ( batch_size == 1 ):
            output[0:3] = output[0:3] + nn.functional.linear(input[i*3:(i+1)*3], wBlock, self.tied_to.bias)
          else:         
            output[:, 0:3] = output[:, 0:3] + nn.functional.linear(input[:, i*3:(i+1)*3], wBlock, self.tied_to.bias)
      
      return output        

    # To keep module properties intuitive
    @property
    def weight(self) -> torch.Tensor:
        return self.tied_to.weight 
        
class blockDecLayer(nn.Module): 
    from torch import Tensor
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(blockDecLayer, self).__init__()
    
        self.in_features = in_features
        self.nIntPts = int(in_features/3)
        self.out_features = out_features
        self.sp = nn.Softplus()
        self.weight = nn.Parameter(torch.empty((self.nIntPts, 6), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(in_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self) -> None:
        import math 
    
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, input: Tensor) -> Tensor:
      #print('Weights ', self.weight)
        
      if ( len(input.shape) == 1 ):
          batch_size = 1
          output = torch.zeros((self.out_features))
      else:
          batch_size = input.shape[0]
          output = torch.zeros((batch_size, self.out_features))
          
      for i in range ( self.nIntPts ):
          wMatrix = torch.zeros((3, 3))

          wMatrix[0, 0] = self.sp(self.weight[i, 0].clone())
          wMatrix[1, 1] = self.sp(self.weight[i, 3].clone())          
          wMatrix[2, 2] = self.sp(self.weight[i, 5].clone()) 
          wMatrix[0, 1] = self.weight[i, 1].clone()
          wMatrix[0, 2] = self.weight[i, 2].clone()
          wMatrix[1, 2] =  self.weight[i, 4].clone()
          
         # print('wmatrix ', wMatrix)
          
          wBlock = torch.matmul(wMatrix.t(), wMatrix)
          
          if ( batch_size == 1 ):
            output[0:3] = output[0:3] + nn.functional.linear(input[i*3:(i+1)*3], wBlock, self.bias)
          else:         
            output[:, 0:3] = output[:, 0:3] + nn.functional.linear(input[:, i*3:(i+1)*3], wBlock, self.bias)
      
      return output
    
    
