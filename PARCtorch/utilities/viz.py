# /utilities/viz.py
import matplotlib.pyplot as plt
import numpy as np
import imageio


def save_gifs_for_channels(
    predictions,
    channels,
    cmaps,
    filename_prefix="predictions",
    interval=0.1,
    batch_idx=0,
):
    """
    Save a sequence of predictions as separate GIFs for each channel with a specified colormap.

    Args:
        predictions (torch.Tensor): The predictions tensor of shape (timesteps, batch_size, channels, height, width).
        channels (list of str): List of channel names (e.g., ['pressure', 'Reynolds', 'u', 'v']).
        cmaps (list of str): List of colormaps for each channel (e.g., ['viridis', 'plasma', 'inferno', 'magma']).
        filename_prefix (str): Prefix for the output GIF filenames.
        interval (float): Time between frames in seconds.
        batch_idx (int): Index of the sample in the batch to visualize.
    """

    # Select the batch sample to visualize
    prediction_sequence = predictions[
        :, batch_idx
    ]  # Shape: (timesteps, channels, height, width)

    # Loop over each channel
    for i, channel_name in enumerate(channels):
        cmap = cmaps[i]  # Get the corresponding colormap for the channel
        frames = []
        for t in range(prediction_sequence.shape[0]):
            # Convert the prediction at timestep t for the current channel to a numpy array
            frame = (
                prediction_sequence[t, i, :, :].cpu().numpy()
            )  # Get the i-th channel

            # Plot the frame and store the image
            fig, ax = plt.subplots()
            cax = ax.imshow(frame, cmap=cmap)  # Use the specified colormap
            fig.colorbar(cax)

            # Save the current frame as an image in memory
            plt.title(f"{channel_name} - Timestep {t+1}")
            plt.axis("off")  # Optional: Turn off axes
            fig.canvas.draw()

            # Convert to a numpy array and store it
            image = np.frombuffer(
                fig.canvas.tostring_rgb(), dtype="uint8"
            ).reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(image)

            # Close the plot to avoid memory issues
            plt.close(fig)

        # Save frames as a gif for the current channel
        gif_filename = f"{filename_prefix}_{channel_name}.gif"
        imageio.mimsave(gif_filename, frames, duration=interval, loop=0)
        print(f"GIF saved to {gif_filename}")


def visualize_channels(
    ic,
    t0,
    t1,
    target,
    channel_names=None,
    channel_cmaps=None,
    sample_index=0,
    batch_size=1,
    figsize=(25, 20),
):
    """
    Visualizes the channels of the initial condition and target timesteps with customizable color maps.

    Args:
        ic (torch.Tensor): Initial condition tensor of shape (batch_size, channels, height, width).
        t0 (torch.Tensor): Scalar tensor (0.0), not used in visualization but kept for compatibility.
        t1 (torch.Tensor): Tensor of shape (future_steps,), not used directly in visualization.
        target (torch.Tensor): Target tensor of shape (future_steps, batch_size, channels, height, width).
        channel_names (list of str, optional): List of channel names for labeling. If None, channels will be unnamed.
        channel_cmaps (list of str or matplotlib.colors.Colormap, optional):
            List of color maps for each channel. If None, 'viridis' is used for all channels.
        sample_index (int, optional): Index of the sample in the batch to visualize. Defaults to 0.
        batch_size (int, optional): Total number of samples in the batch. Needed if `sample_index` is to be visualized.
        figsize (tuple, optional): Figure size for the plots. Defaults to (25, 20).

    Raises:
        ValueError: If `sample_index` is out of bounds or if `channel_cmaps` length doesn't match number of channels.
    """
    # Validate sample_index
    if sample_index >= batch_size or sample_index < 0:
        raise ValueError(
            f"sample_index {sample_index} is out of bounds for batch size {batch_size}."
        )

    # Select the specific sample from the batch
    ic_sample = ic[sample_index]  # Shape: (channels, height, width)
    target_sample = target[
        :, sample_index
    ]  # Shape: (future_steps, channels, height, width)

    num_channels = ic_sample.shape[0]
    future_steps = target_sample.shape[0]

    # Handle channel_cmaps
    if channel_cmaps is None:
        # Use 'viridis' for all channels by default
        channel_cmaps = ["viridis"] * num_channels
    else:
        if not isinstance(channel_cmaps, list):
            raise TypeError(
                "channel_cmaps must be a list of color map names or Colormap objects."
            )
        if len(channel_cmaps) != num_channels:
            raise ValueError(
                f"Length of channel_cmaps ({len(channel_cmaps)}) does not match number of channels ({num_channels}). "
                f"Ensure you provide a color map for each channel or leave it as None to use default."
            )

    # Debugging: Print min and max of each channel
    print("Channel Data Statistics:")
    for idx in range(num_channels):
        channel_data_ic = ic_sample[idx].cpu().numpy()
        channel_data_target = target_sample[:, idx].cpu().numpy()
        print(
            f"Channel {idx}: IC min={channel_data_ic.min()}, IC max={channel_data_ic.max()}"
        )
        for step in range(future_steps):
            print(
                f"  Step {step + 1}: min={channel_data_target[step].min()}, max={channel_data_target[step].max()}"
            )

    # Determine subplot grid size
    cols = num_channels
    rows = future_steps + 1  # +1 for the initial condition

    # Create a figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Handle axes array shape
    if num_channels == 1:
        axes = axes.reshape(-1, 1)
    if rows == 1:
        axes = axes.reshape(1, -1)

    # Plot Initial Condition
    for channel in range(num_channels):
        ax = axes[0, channel]
        data = ic_sample[channel].cpu().numpy()
        cmap = channel_cmaps[channel]
        im = ax.imshow(data, cmap=cmap)
        if channel_names:
            ax.set_title(f"IC - {channel_names[channel]}")
        else:
            ax.set_title(f"IC - Channel {channel}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Plot Targets
    for step in range(future_steps):
        for channel in range(num_channels):
            ax = axes[step + 1, channel]
            data = target_sample[step, channel].cpu().numpy()
            cmap = channel_cmaps[channel]
            im = ax.imshow(data, cmap=cmap)
            if channel_names:
                ax.set_title(f"Step {step + 1} - {channel_names[channel]}")
            else:
                ax.set_title(f"Step {step + 1} - Channel {channel}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


# Visualization Function
def save_gifs_with_ground_truth(
    predictions,
    ground_truth,
    channels,
    cmaps,
    filename_prefix="comparison",
    interval=0.1,
    batch_idx=0,
):
    """
    Save a sequence of predictions and ground truth as GIFs for each channel.

    Args:
        predictions (torch.Tensor): Predictions tensor of shape (timesteps, batch_size, channels, height, width).
        ground_truth (torch.Tensor): Ground truth tensor of shape (timesteps, batch_size, channels, height, width).
        channels (list of str): Channel names.
        cmaps (list of str): Colormaps for each channel.
        filename_prefix (str): Prefix for GIF filenames.
        interval (float): Time between frames.
        batch_idx (int): Batch index to visualize.
    """

    prediction_sequence = predictions[
        :, batch_idx
    ].cpu()  # Shape: (timesteps, channels, height, width)
    ground_truth_sequence = ground_truth[
        :, batch_idx
    ].cpu()  # Shape: (timesteps, channels, height, width)

    for i, channel_name in enumerate(channels):
        cmap = cmaps[i]
        frames = []
        for t in range(prediction_sequence.shape[0]):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            # Prediction
            ax = axes[0]
            pred_frame = prediction_sequence[t, i].numpy()
            im = ax.imshow(pred_frame, cmap=cmap)
            ax.set_title(f"Predicted - Timestep {t+1}")
            ax.axis("off")
            fig.colorbar(im, ax=ax)

            # Ground Truth
            ax = axes[1]
            gt_frame = ground_truth_sequence[t, i].numpy()
            im = ax.imshow(gt_frame, cmap=cmap)
            ax.set_title(f"Ground Truth - Timestep {t+1}")
            ax.axis("off")
            fig.colorbar(im, ax=ax)

            fig.canvas.draw()
            image = np.frombuffer(
                fig.canvas.tostring_rgb(), dtype="uint8"
            ).reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(image)
            plt.close(fig)

        gif_filename = f"{filename_prefix}_{channel_name}.gif"
        imageio.mimsave(gif_filename, frames, duration=interval, loop=0)
        print(f"GIF saved to {gif_filename}")
