import torch
import os


def load_model_weights(model, weights_path, device):
    """
    Loads the saved weights into the model.

    Args:
        model (torch.nn.Module): The model instance.
        weights_path (str): Path to the saved model weights (.pth file).
        device (torch.device): Device to load the model onto.

    Returns:
        model (torch.nn.Module): The model with loaded weights.
    """
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Model weights file not found at '{weights_path}'"
        )

    # Load the state dictionary
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded model weights from '{weights_path}'")

    # Move the model to the specified device
    model = model.to(device)

    # Set the model to evaluation mode
    model.eval()

    return model


