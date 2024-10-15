#!/usr/bin/env python3
"""
Training Script for PARCTorch Model

This script performs the following steps:
1. Data Normalization
2. Dataset and DataLoader Preparation
3. Model Building
4. Training Loop with Logging and Checkpointing
5. Saving Loss History

Usage:
    python train_parc.py --config config.yaml

Author: Your Name
Date: 2024-10-10
"""

import os
import sys
import argparse
import logging
import pickle
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------
# Add PARCTorch to system path
# ---------------------------
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

# ---------------------------
# Import PARCTorch Modules
# ---------------------------
from data.normalization import compute_min_max
from data.dataset import GenericPhysicsDataset, custom_collate_fn
from utilities.viz import visualize_channels
from PARCtorch.PARCv2 import PARCv2
from PARCtorch.differentiator.differentiator import Differentiator
from PARCtorch.differentiator.finitedifference import FiniteDifference
from PARCtorch.integrator.integrator import Integrator
from PARCtorch.integrator.heun import Heun
from PARCtorch.utilities.unet import UNet


# ---------------------------
# Argument Parsing
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train PARCTorch Model")
    parser.add_argument(
        "--train_dirs",
        nargs="+",
        required=True,
        help="List of training data directories",
    )
    parser.add_argument(
        "--min_max_output",
        type=str,
        default="../data/ns_min_max.json",
        help="Path to save min and max normalization values",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="../weights",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--loss_dir",
        type=str,
        default="../loss",
        help="Directory to save loss history",
    )
    parser.add_argument(
        "--log_file", type=str, default="training.log", help="Log file name"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize a batch before training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training (cuda or cpu)",
    )
    return parser.parse_args()


# ---------------------------
# Setup Logging
# ---------------------------
def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )


# ---------------------------
# Data Normalization
# ---------------------------
def perform_normalization(data_dirs, output_file, device):
    logging.info("Starting data normalization...")
    compute_min_max(data_dirs, output_file)
    logging.info(f"Min and max values saved to '{output_file}'.")


# ---------------------------
# Create DataLoader
# ---------------------------
def create_dataloader(train_dir, min_max_path, batch_size, device):
    logging.info("Preparing DataLoader...")
    train_dataset = GenericPhysicsDataset(
        data_dirs=[train_dir], future_steps=1, min_max_path=min_max_path
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Shuffled for training
        num_workers=1,  # Adjust based on your system
        pin_memory=True if device == "cuda" else False,
        collate_fn=custom_collate_fn,
    )

    logging.info(f"Total samples in training dataset: {len(train_dataset)}")
    return train_loader


# ---------------------------
# Visualize Data
# ---------------------------
def visualize_data(train_loader, device):
    logging.info("Visualizing a batch from the DataLoader...")
    # model = None  # Not needed for visualization
    for batch in train_loader:
        ic, t0, t1, target = batch
        channel_names = [
            "Pressure (P)",
            "Reynolds (R)",
            "Velocity U",
            "Velocity V",
        ]
        custom_cmaps = ["seismic", "seismic", "seismic", "seismic"]

        visualize_channels(
            ic,
            t0,
            t1,
            target,
            channel_names=channel_names,
            channel_cmaps=custom_cmaps,
        )
        break  # Visualize only the first batch
    logging.info("Data visualization completed.")


# ---------------------------
# Build Model
# ---------------------------
def build_model(device):
    logging.info("Building the PARCTorch model...")
    # Define model parameters
    n_fe_features = 128
    unet_ns = UNet(
        [64, 128, 256, 512, 1024],  # Feature maps
        in_channels=4,  # Number of input channels
        out_channels=n_fe_features,
        up_block_use_concat=[False, True, False, True],
        skip_connection_indices=[2, 0],
    ).to(device)

    right_diff = FiniteDifference(padding_mode="replicate").to(device)

    heun_int = Heun().to(device)

    diff_ns = Differentiator(
        n_state_vars=2,  # p and re
        n_fe_features=n_fe_features,
        advection_channels=[2, 3],  # u and v
        diffusion_channels=[2, 3],  # u and v
        feature_extractor=unet_ns,
        padding_mode="constant",
        finite_diff=right_diff,
    ).to(device)

    ns_int = Integrator(
        clip_input=True,
        poisson_indices=[(0, 2, 3, 1)],  # (p, u, v, re)
        integrator=heun_int,
        data_driven_integrator=[None, None, None, None],
        padding_mode="constant",
        finite_diff=right_diff,
    ).to(device)

    criterion = torch.nn.L1Loss().to(device)

    model = PARCv2(
        differentiator=diff_ns, integrator=ns_int, criterion=criterion
    ).to(device)

    optimizer = Adam(model.parameters(), lr=1e-5)

    logging.info("Model built successfully.")
    return model, optimizer, criterion


# ---------------------------
# Training Loop
# ---------------------------
def train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    num_epochs,
    save_dir,
    loss_dir,
    device,
):
    logging.info("Starting training...")

    # Ensure directories exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(loss_dir, exist_ok=True)

    epoch_losses = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{num_epochs}",
        )

        for batch_idx, batch in progress_bar:
            ic, t0, t1, gt = batch

            # Move data to device
            ic = ic.to(device, non_blocking=True)
            t0 = t0.to(device, non_blocking=True)
            t1 = t1.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass
            predictions = model(ic, t0, t1)

            # Compute loss
            loss = criterion(predictions[:, :, 1:, :, :], gt[:, :, 1:, :, :])

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({"Batch Loss": f"{loss.item():.6f}"})

        # Average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)

        logging.info(
            f"Epoch [{epoch}/{num_epochs}], Average Loss: {epoch_loss:.7f}"
        )

        # Save model checkpoint every 10 epochs
        if epoch % 10 == 0 or epoch == num_epochs:
            model_save_path = os.path.join(
                save_dir, f"ns_model_epoch_{epoch}.pth"
            )
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"Model checkpoint saved at '{model_save_path}'")

    # Save loss history
    loss_save_path = os.path.join(loss_dir, "ns_losses.pkl")
    with open(loss_save_path, "wb") as f:
        pickle.dump(epoch_losses, f)
    logging.info(f"Loss history saved at '{loss_save_path}'")

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker="o")
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.grid(True)
    loss_plot_path = os.path.join(loss_dir, "loss_curve.png")
    plt.savefig(loss_plot_path)
    plt.close()
    logging.info(f"Loss curve saved at '{loss_plot_path}'")

    logging.info("Training completed successfully.")


# ---------------------------
# Main Function
# ---------------------------
def main():
    args = parse_args()

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.loss_dir, f"{timestamp}_{args.log_file}")
    os.makedirs(args.loss_dir, exist_ok=True)
    setup_logging(log_file)

    logging.info("Training script started.")

    device = (
        args.device
        if torch.cuda.is_available() and args.device == "cuda"
        else "cpu"
    )
    logging.info(f"Using device: {device}")

    # Data Normalization
    perform_normalization(args.train_dirs, args.min_max_output, device)

    # Create DataLoader
    train_loader = create_dataloader(
        args.train_dirs[0], args.min_max_output, args.batch_size, device
    )

    # Optional Visualization
    if args.visualize:
        visualize_data(train_loader, device)

    # Build Model
    model, optimizer, criterion = build_model(device)

    # Start Training
    train_model(
        model,
        train_loader,
        criterion,
        optimizer,
        args.num_epochs,
        args.save_dir,
        args.loss_dir,
        device,
    )

    logging.info("Training script finished.")


if __name__ == "__main__":
    main()
