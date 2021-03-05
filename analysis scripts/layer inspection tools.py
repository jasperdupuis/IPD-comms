# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:24:21 2021

@author: Jasper
"""

inputs = []
hiddens_1 = []
outputs = []


for net in nets:
    inputs.append(net._modules['0'])
    hiddens_1.append(net._modules['2'])
    outputs.append(net._modules['4'])
    
input_w = []    
input_b = []
for layer in inputs:
    input_w.append(layer.weight.data.numpy())
    input_b.append(layer.bias.data.numpy())

output_w = []    
output_b = []
for layer in outputs:
    output_w.append(layer.weight.data.numpy())
    output_b.append(layer.bias.data.numpy())
    
x = input_w[0]
np.mean(x)
np.std(x)
