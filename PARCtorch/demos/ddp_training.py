import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import pickle

# Add the root directory (PARCTorch) to the system path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from data.multigpu_dataset import MultiGPUGenericPhysicsDataset, multi_gpu_collate_fn
sys.path.append("../../")
from PARCtorch.PARCv2 import PARCv2
from PARCtorch.differentiator.differentiator import Differentiator
from PARCtorch.differentiator.finitedifference import FiniteDifference
from PARCtorch.integrator.integrator import Integrator
from PARCtorch.integrator.heun import Heun
from PARCtorch.utilities.unet import UNet

def ddp_setup(rank, world_size):
    # Initialize the process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size, data_dir_train, data_dir_test, min_max_path, batch_size, num_epochs, save_dir, loss_dir):
    ddp_setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Initialize your model components
    n_fe_features = 64
    unet_burgers = UNet([64, 64*2, 64*4], 3, n_fe_features, up_block_use_concat=[False, True], skip_connection_indices=[0]).to(device)
    right_diff = FiniteDifference(padding_mode="replicate").to(device)
    heun_int = Heun().to(device)

    diff_burgers = Differentiator(
        1,                 # 1 state variable
        n_fe_features,     # Number of features from UNet: 64
        [1, 2],            # Channels for advection: u and v
        [1, 2],            # Channels for diffusion: u and v
        unet_burgers,
        "constant",
        right_diff
    ).to(device)

    burgers_int = Integrator(True, [], heun_int, [None, None, None], "constant", right_diff).to(device)
    criterion = nn.L1Loss().to(device)
    model = PARCv2(diff_burgers, burgers_int, criterion).to(device)

    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])

    # Create Dataset and DataLoader with DistributedSampler
    train_dataset_mgpu = MultiGPUGenericPhysicsDataset(
        data_dirs=[data_dir_train],
        future_steps=1,
        min_max_path=min_max_path
    )
    train_sampler = DistributedSampler(train_dataset_mgpu, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_dataset_mgpu,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=multi_gpu_collate_fn
    )

    # Initialize Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # Training loop
    epoch_losses = []
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)  # Set epoch for shuffling data
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{num_epochs}")

        for batch_idx, batch in progress_bar:
            ic, t0, t1, gt = batch
            ic, t0, t1, gt = ic.to(device), t0.to(device), t1.to(device), gt.to(device)

            optimizer.zero_grad()
            predictions = model(ic, t0, t1)
            predictions = predictions.permute(1, 0, 2, 3, 4)

            loss = criterion(predictions[:, :, 1:, :, :], gt[:, :, 1:, :, :])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({"Batch Loss": loss.item()})

        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)
        print(f"Epoch [{epoch}/{num_epochs}], Average Loss: {epoch_loss:.7f}")

        if rank == 0 and epoch % 10 == 0:
            model_save_path = os.path.join(save_dir, f"ddp_model_epoch_{epoch}.pth")
            torch.save(model.module.state_dict(), model_save_path)
            print(f"Model weights saved at {model_save_path}")

    if rank == 0:
        loss_save_path = os.path.join(loss_dir, 'ddp_losses.pkl')
        with open(loss_save_path, 'wb') as f:
            pickle.dump(epoch_losses, f)
        print(f"Losses saved at {loss_save_path}")

    cleanup()

# Spawn processes for each GPU
def run_ddp_training(world_size, data_dir_train, data_dir_test, min_max_path, batch_size=8, num_epochs=50, save_dir='../weights', loss_dir='../loss'):
    mp.spawn(
        train_ddp,
        args=(world_size, data_dir_train, data_dir_test, min_max_path, batch_size, num_epochs, save_dir, loss_dir),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    # Define the dataset paths and parameters
    data_dir_train = '/project/vil_baek/data/physics/PARCTorch/Burgers/train'
    data_dir_test = '/project/vil_baek/data/physics/PARCTorch/Burgers/test'
    min_max_path = '../data/b_min_max.json'

    # Set the MASTER_ADDR and MASTER_PORT
    os.environ['MASTER_ADDR'] = 'localhost'  # Replace with master node IP if multi-node
    os.environ['MASTER_PORT'] = '12355'      # Ensure this port is free

    # Check the number of GPUs
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise ValueError("No GPUs available for training.")

    # Start DDP training
    run_ddp_training(world_size, data_dir_train, data_dir_test, min_max_path)
