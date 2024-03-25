from networks.network import Network
from networks.network_utils import get_inplace_acti, pytorch_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

BIAS = True

def create_layers(inp_dim, out_dim, activation='none', norm=False, use_bias=True):
    if activation == 'crelu': out_dim = int(out_dim / 2) 
    layer = [nn.Linear(inp_dim, out_dim, bias=use_bias)]
    if norm: layer = [nn.LayerNorm(inp_dim)] + layer
    layer = layer + [get_inplace_acti(activation)]
    return layer

class MLPNetwork(Network):    
    def __init__(self, args):
        super().__init__(args)
        self.scale_final = args.scale_final
        self.is_crelu = args.activation == "crelu"
        if args.activation_final == "crelu":
            self.activation_final = get_inplace_acti("leakyrelu")
        sizes = [self.num_inputs] + self.hs + [self.num_outputs]
        activations = [args.activation for i in range(len(sizes)-2)] + ['none'] # last layer is none
        layers = list()
        for inp_dim, out_dim, acti in zip(sizes, sizes[1:], activations):
            layers += create_layers(int(inp_dim), int(out_dim), activation=acti, norm = self.use_layer_norm, use_bias=args.use_bias)
        if args.dropout > 0: # only supports last layer dropout TODO: for now
            layers = layers[:-1] + [nn.Dropout(args.dropout)] + [layers[-1]]
        self.model = nn.Sequential(*layers)
        self.train()
        self.reset_network_parameters()

    def forward(self, x):
        if type(x) != torch.tensor: x = pytorch_model.wrap(x, cuda =self.iscuda, device = self.device)
        x = self.model(x)
        x = self.activation_final(x)
        x = x * self.scale_final
        return x