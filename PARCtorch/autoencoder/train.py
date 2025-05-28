import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from autoencoder import *
from utils import * 

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
    

def train_autoencoder(model, optimizer, loss_function, train_loader, val_loader, 
                      device, epochs=10, image_size=(64, 64), n_channels=3, 
                      scheduler=None, noise_fn=None, initial_max_noise=0.16, 
                      n_reduce_factor=0.5, reduce_on=1000, save_path=None, weights_name=None):
    """ Train an autoencoder with optional noise injection. """

    log_dict = {'training_loss_per_epoch': [], 'validation_loss_per_epoch': []}
    
    model.to(device)

    max_noise = initial_max_noise  # Initial noise level

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Reduce noise periodically
        if (epoch + 1) % reduce_on == 0:
            max_noise *= n_reduce_factor

        # --- Training ---
        model.train()
        train_losses = []
        for images in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            images = images[0][:, 0:n_channels, ...].to(device)
            
            # Apply noise if function is provided
            noisy_images = noise_fn(images, max_val=max_noise) if noise_fn else images

            # Forward pass
            output = model(noisy_images)
            loss = loss_function(output, images.view(-1, n_channels, *image_size))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        log_dict['training_loss_per_epoch'].append(avg_train_loss)

        # --- Validation ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for val_images in tqdm(val_loader, desc="Validating"):
                val_images = val_images[0][:, 0:n_channels, ...].to(device)
                
                # val_images = F.interpolate(val_images, size=(image_size[0], image_size[1]), mode='bilinear', align_corners=False) #!!! for MLP

                # Forward pass
                output = model(val_images)
                val_loss = loss_function(output, val_images.view(-1, n_channels, *image_size))
                val_losses.append(val_loss.item())

        avg_val_loss = np.mean(val_losses)
        log_dict['validation_loss_per_epoch'].append(avg_val_loss)

        print(f"Epoch {epoch+1}: Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")

        if scheduler:
            scheduler.step(avg_val_loss)

    # Save model and logs
    save_model(model, save_path, weights_name, epochs)
    save_log(log_dict, save_path, weights_name, epochs)

    return log_dict


def train_individual_autoencoder(model, optimizer, loss_function, train_loader, val_loader, 
                      device, epochs=10, image_size=(64, 64), channel_index=0, 
                      scheduler=None, noise_fn=None, initial_max_noise=0.16, 
                      n_reduce_factor=0.8, reduce_on=1000, save_path=None, weights_name=None):
    
    """ Train an autoencoder on just one channel at a time with optional noise injection. """

    log_dict = {'training_loss_per_epoch': [], 'validation_loss_per_epoch': []}
    
    model.to(device)

    max_noise = initial_max_noise  # Initial noise level

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Reduce noise periodically
        if (epoch + 1) % reduce_on == 0:
            max_noise *= n_reduce_factor

        # --- Training ---
        model.train()
        train_losses = []
        for images in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            images = images[0][:, channel_index:channel_index+1, ...].to(device)
            
            # Apply noise if function is provided
            noisy_images = noise_fn(images, max_val=max_noise) if noise_fn else images

            # Forward pass
            output = model(noisy_images)
            loss = loss_function(output, images.view(-1, 1, *image_size))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        log_dict['training_loss_per_epoch'].append(avg_train_loss)

        # --- Validation ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for val_images in tqdm(val_loader, desc="Validating"):
                val_images = val_images[0][:, channel_index:channel_index+1, ...].to(device)

                # Forward pass
                output = model(val_images)
                val_loss = loss_function(output, val_images.view(-1, 1, *image_size))
                val_losses.append(val_loss.item())

        avg_val_loss = np.mean(val_losses)
        log_dict['validation_loss_per_epoch'].append(avg_val_loss)

        print(f"Epoch {epoch+1}: Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")

        if scheduler:
            scheduler.step(avg_val_loss)

    # Save model and logs
    save_model(model, save_path, weights_name, epochs)
    save_log(log_dict, save_path, weights_name, epochs)

    return log_dict