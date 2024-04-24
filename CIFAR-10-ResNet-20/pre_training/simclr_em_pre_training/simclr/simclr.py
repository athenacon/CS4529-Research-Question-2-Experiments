import torch.nn as nn
import torchvision
from simclr.modules.identity import Identity

class SimCLR(nn.Module):

    def __init__(self, encoder, caps_net):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        self.caps_net = caps_net

        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()
        self.encoder.avgpool = Identity()


    def forward(self, x_i, x_j):     
        
        x_i = self.encoder(x_i)
        x_j = self.encoder(x_j)
        
        z_i = self.caps_net(x_i)
        z_j = self.caps_net(x_j)
        return z_i, z_j
        