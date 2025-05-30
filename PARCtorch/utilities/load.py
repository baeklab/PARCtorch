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
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Loaded model weights from '{weights_path}'")

    # Move the model to the specified device
    model = model.to(device)

    # Set the model to evaluation mode
    model.eval()

    return model

def get_device():
    """Returns the available device: CUDA, MPS or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
def resolve_device(device=None):
    """
    Resolves the device based on input or availability.

    Args:
        device (str or torch.device, optional): Preferred device string or object.

    Returns:
        torch.device: The resolved device (e.g., 'cuda', 'mps', or 'cpu').
    """
    if device is None:
        return get_device()
    return torch.device(device)


