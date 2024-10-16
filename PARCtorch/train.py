import os
import torch
from tqdm import tqdm  # Import tqdm for progress bar

# Training loop with tqdm for progress bars
def train_model(model, train_loader, criterion, optimizer, num_epochs, save_dir):
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = model.cuda()  # Move model to GPU
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
            ic = ic.cuda()
            t0 = t0.cuda()
            t1 = t1.cuda()
            gt = gt.cuda()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            predictions = model(ic, t0, t1)

            # Compute loss
            loss = criterion(predictions[:, :, 1:, :, :], gt[:, :, 1:, :, :])

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate running loss
            running_loss += loss.item()

            # Update tqdm progress bar with current loss
            progress_bar.set_postfix({"Batch Loss": loss.item()})

        # Calculate and print average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch}/{num_epochs}], Average Loss: {epoch_loss:.4f}")

        # Save the model weights at the end of the epoch
        model_save_path = os.path.join(save_dir, "model.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model weights saved at {model_save_path}")
