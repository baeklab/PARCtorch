import os
import torch
from tqdm import tqdm
import pickle  # Import pickle to save the loss
from PARCtorch.utilities.load import resolve_device

# Training loop with tqdm for progress bars
def train_model(model, train_loader, criterion, optimizer, num_epochs, save_dir, app):

    # Device selection
    device = resolve_device()

    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = model.to(device)  # Move model to GPU
    all_losses = []  # List to store epoch losses

    for epoch in range(1, num_epochs + 1):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        # Add tqdm progress bar to track the training batches
        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{num_epochs}",
        )

        for batch_idx, batch in progress_bar:
            # Assuming batch contains ic, t0, t1, and ground truth (gt)
            ic, t0, t1, gt = batch  # Unpack the batch tuple

            # Move data to GPU
            ic = ic.to(device) 
            t0 = t0.to(device) 
            t1 = t1.to(device) 
            gt = gt.to(device) 

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            predictions = model(ic, t0, t1)

            # Compute loss based on the 'app' parameter
            # If else logic for the application type burgers, navier-stokes (ns), or energetic materials (em)
            # This logic is explicitly not incuding the reynolds uses in burgers and ns
            if app == "burgers":
                # our uploaded burgers data has reynolds in the first channel (index 0 ),
                loss = criterion(predictions[:, :, 1:, :, :], gt[:, :, 1:, :, :])
            elif app == "ns":
                # Skip channel at index 1
                # our uploaded ns data has reynolds in the second channel (index 1),
                loss = criterion(
                    torch.cat((predictions[:, :, :1, :, :], predictions[:, :, 2:, :, :]), dim=2),
                    torch.cat((gt[:, :, :1, :, :], gt[:, :, 2:, :, :]), dim=2)
                )
            elif app == "em":
                # we do not have reynolds number in our uploaded em data so we can use all channels
                # the 0 index is the first channel and coded just to follow the convention above
                # ie all channles are used
                loss = criterion(predictions[:, :, 0:, :, :], gt[:, :, 0:, :, :])
            else:
                raise ValueError(f"Unknown application type: {app}")

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate running loss
            running_loss += loss.item()

            # Update tqdm progress bar with current loss
            progress_bar.set_postfix({"Batch Loss": loss.item()})

        # Calculate and print average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        all_losses.append(epoch_loss)  # Store epoch loss
        print(f"Epoch [{epoch}/{num_epochs}], Average Loss: {epoch_loss:.4f}")

        # Save the model weights at the end of the epoch
        model_save_path = os.path.join(save_dir, "model.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model weights saved at {model_save_path}")

    # Save all epoch losses to a pickle file
    loss_save_path = os.path.join(save_dir, "training_losses.pkl")
    with open(loss_save_path, "wb") as f:
        pickle.dump(all_losses, f)
    print(f"Training losses saved at {loss_save_path}")
