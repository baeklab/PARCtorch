import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
import json
import torch.nn.functional as F
    
# Define Modules #

class MLPEncoder(nn.Module):
    def __init__(self, layers, latent_dim, act_fn=nn.ReLU()):
        '''
        note: layers and latent dim are treated the same as for convolutional autoencoder at the input level, but under the hood the MLP model flattens each layer as it should

        layers: list, channel values excluding latent dim channels e.g [3, 8], should be the same as layers for decoder
        latent_dim: int, number of channels to have in the latent bottlneck layer
        act_fn: activation function to be used throughout entire model
        '''
        super().__init__()
        modules = []
        in_dim = layers[0]
        for dim in layers[1:]:
            modules.append(nn.Linear(in_dim, dim))
            modules.append(act_fn)
            in_dim = dim
        modules.append(nn.Linear(in_dim, latent_dim))  # Bottleneck layer
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        # Flatten input except batch dimension
        x = x.view(x.size(0), -1)
        return self.net(x)

    
class MLPDecoder(nn.Module):
    def __init__(self, layers, latent_dim, output_shape=(3, 128, 256), act_fn=nn.ReLU()):
        '''
        note: layers and latent dim are treated the same as for convolutional autoencoder at the input level, but under the hood the MLP model flattens each layer as it should

        layers: list, channel values excluding latent dim channels e.g [3, 8], should be the same as layers for decoder
        latent_dim: int, number of channels to have in the latent bottlneck layer
        output_shape: tuple, used to reshape the flattened vector correctly upon output (n_channels, height, width)
        act_fn: activation function to be used throughout entire model
        '''
        super().__init__()
        self.output_shape = output_shape  
        modules = []
        in_dim = latent_dim
        for dim in reversed(layers):
            modules.append(nn.Linear(in_dim, dim))
            modules.append(act_fn)
            in_dim = dim
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        x = self.net(x)
        batch_size = x.size(0)
        return x.view(batch_size, *self.output_shape)
    

# Convolutional AE
class Encoder(nn.Module):
    def __init__(self, layers, latent_dim=16, act_fn=nn.ReLU()):
        '''
        layers: list, channel values excluding latent dim channels e.g [3, 8], should be the same as layers for decoder
        latent_dim: int, number of channels to have in the latent bottlneck layer
        act_fn: activation function to be used throughout entire model
        '''
        super().__init__()
        modules = []
        in_channels = layers[0]
        for out_channels in layers[1:]:
            modules.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            )  # Keep padding=1 for same-sized convolutions
            modules.append(act_fn)
            in_channels = out_channels
        modules.append(
            nn.Conv2d(layers[-1], latent_dim, kernel_size=3, stride = 2, padding=1)
        )  # Bottleneck layer
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):    # no deconv
    def __init__(self, layers, latent_dim=16, act_fn=nn.ReLU()):
        '''
        layers: list, channel values excluding latent dim channels e.g [3, 8], should be the same as layers for encoder
        latent_dim: int, number of channels to have in the latent bottlneck layer
        act_fn: activation function to be used throughout entire model
        '''
        super().__init__()

        self.in_channels = layers[-1]
        self.latent_dim = latent_dim

        modules = []
        in_channels = latent_dim 

        # Iteratively create resize-convolution layers
        for out_channels in reversed(layers):
            modules.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))  # Resizing
            modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))  # Convolution
            modules.append(act_fn)  # Activation function
            in_channels = out_channels
            
        # modules.pop() # final activation linear
        
        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        return self.conv(x)


# Defining the full autoencoder

class Autoencoder(nn.Module):
    '''
    Wrapper for autoencoder with 1 encoder and 1 decoder handling all data channels together
    '''
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class AutoencoderSeparate(nn.Module):
    def __init__(self, encoder_T, encoder_P, encoder_M, decoder_T, decoder_P, decoder_M):
        '''
        Wrapper for autoencoder with 3 encoders and 3 decoders handling all data channels separate. Right now this is hard coded to be just for 3 channels corresponding to EM data (though any data with 3 channels would be compatible).
        '''
        super().__init__()
        self.encoderT = encoder_T
        self.encoderP = encoder_P
        self.encoderM = encoder_M
        self.decoderT = decoder_T
        self.decoderP = decoder_P
        self.decoderM = decoder_M
    
    def forward(self, x):
        z_t = self.encoderT(x[:, 0:1, :, :]) # only T channel
        z_p = self.encoderP(x[:, 1:2, :, :]) # only P channel
        z_m = self.encoderM(x[:, 2:3, :, :]) # only M channel
                
        decoded_t = self.decoderT(z_t) # decode T
        decoded_p = self.decoderP(z_p) # decode P
        decoded_m = self.decoderM(z_m) # decode M
        decoded = torch.cat((decoded_t, decoded_p, decoded_m), dim=1) # concat for output
        
        return decoded
    
class ConvolutionalAutoencoder:
    def __init__(self, autoencoder, optimizer, device, save_path=None, weights_name=None):
        self.network = autoencoder.to(device)
        self.optimizer = optimizer
        self.device = device
        self.save_path = save_path
        self.weights_name = weights_name

    def autoencode(self, x):
        return self.network(x)

    def encode(self, x):
        return self.network.encoder(x)

    def decode(self, x):
        return self.network.decoder(x)
