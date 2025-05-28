import torch
import json

def save_model(model, save_path, weights_name, epochs):
    """ Save model weights to file. """
    if save_path:
        save_file = f"{save_path}/{weights_name}_{epochs}.pth"
        torch.save(model.state_dict(), save_file)
        print(f"Model weights saved to {save_file}")

def save_log(log_dict, save_path, weights_name, epochs):
    """ Save training logs as JSON. """
    if save_path:
        log_file = f"{save_path}/{weights_name}_{epochs}.json"
        with open(log_file, 'w') as f:
            json.dump(log_dict, f)
            
def add_random_noise(images, min_val=0.0, max_val=0.1):
    """
    Add random (uniform) noise to the images.

    Parameters:
        images: Tensor of input images.
        min_val: Minimum value of the noise.
        max_val: Maximum value of the noise.

    Returns:
        Noisy images.
    """
    noise = torch.rand_like(images) * (max_val - min_val) + min_val
    noisy_images = images + noise
    return torch.clamp(noisy_images, 0.0, 1.0)  # Keep pixel values in [0, 1]

class LpLoss(torch.nn.Module):
    def __init__(self, p=10):
        super(LpLoss, self).__init__()
        self.p = p

    def forward(self, input, target):
        # Compute element-wise absolute difference
        diff = torch.abs(input - target)
        # Raise the differences to the power of p, sum them, and raise to the power of 1/p
        return (torch.sum(diff ** self.p) ** (1 / self.p))