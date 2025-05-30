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

class LpLoss(nn.Module):
    def __init__(self, p=10, reduction='mean'):
        super(LpLoss, self).__init__()
        self.p = p
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.reduction = reduction

    def forward(self, input, target):
        diff = torch.abs(input - target) ** self.p
        loss = torch.sum(diff, dim=tuple(range(1, diff.ndim))) ** (1 / self.p)  # norm per sample

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:  # 'none'
            return loss
