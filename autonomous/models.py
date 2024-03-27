import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

def resnet(num_outputs, num_input_channels=3, output_activation=nn.Identity, pretrained=True):
    model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=pretrained)
    model.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, num_outputs),
        output_activation()
    )
    return model
    

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, device):
        super().__init__()
        self.device = device
        self.pi_mean = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, nn.Tanh)
        self.pi_logstd = nn.Parameter(torch.zeros(act_dim))
        self.v = mlp([obs_dim] + list(hidden_sizes) + [1], activation, nn.Identity)

    def compute(self, x, probablistic=True):
        pi_mean = self.pi_mean(x)
        pi_logstd = self.pi_logstd.expand_as(pi_mean)
        v = self.v(x).squeeze(1)
        if probablistic:
            std = torch.exp(pi_logstd)
            dist = torch.distributions.Normal(pi_mean, std)
            return dist, v
        return pi_mean, v
    
    def forward(self, obs):
        with torch.no_grad():
            dist, _ = self.compute(obs, probablistic=True)
            a = dist.sample()
            a = torch.clamp(a, -1, 1)
            return a.cpu().numpy()
        
    def log_prob(self, obs, act):
        dist, _ = self.compute(obs, probablistic=True)
        return dist.log_prob(act).sum(axis=-1)
    
    def entropy(self, obs):
        dist, _ = self.compute(obs, probablistic=True)
        return dist.entropy().sum(axis=-1)
