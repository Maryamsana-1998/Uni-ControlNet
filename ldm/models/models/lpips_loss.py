import torch
import lpips

class LPIPSLoss(torch.nn.Module):
    """Calculate LPIPS loss with optional standardization."""

    def __init__(self, net='vgg', verbose=False, standardize=False, mean=None, std=None):
        super(LPIPSLoss, self).__init__()
        print('lpips loss with ',net)
        self.lpips_func = lpips.LPIPS(net=net, verbose=verbose, lpips=True)
        self.standardize = standardize
        self.mean = mean
        self.std = std

        # Ensure LPIPS model parameters are not updated during training
        for param in self.lpips_func.parameters():
            param.requires_grad = False

    def forward(self, fake_image, real_image):
        """Assuming inputs are in [0, 255] range."""
        # print('shape:', fake_image.shape)
        if self.standardize:
            # Standardize images with provided or computed mean and std
            fake_image = self._standardize_image(fake_image)
            real_image = self._standardize_image(real_image)
        else:
            # If not standardizing, normalize by dividing by 255
            fake_image = fake_image / 255.0 * 2 - 1.0
            real_image = real_image / 255.0 * 2 - 1.0

        # Compute LPIPS loss
        loss = self.lpips_func(fake_image, real_image)
        return loss.mean()  # Loss is per image in the batch

    def _standardize_image(self, image):
        # Compute mean and std if not provided
        if self.mean is None:
            mean = image.mean([0, 2, 3], keepdim=True)  # Compute mean over batch and spatial dimensions
        else:
            mean = torch.tensor(self.mean, device=image.device).view(1, -1, 1, 1)

        if self.std is None:
            std = image.std([0, 2, 3], keepdim=True)  # Compute std over batch and spatial dimensions
        else:
            std = torch.tensor(self.std, device=image.device).view(1, -1, 1, 1)

        # Standardize the image
        image_standardized = (image - mean) / (std + 1e-8)  # Add epsilon to avoid division by zero

        # Normalize to [-1, 1] range
        min_val = image_standardized.min(dim=-1, keepdim=True)[0]
        max_val = image_standardized.max(dim=-1, keepdim=True)[0]
        image_normalized = 2 * (image_standardized - min_val) / (max_val - min_val + 1e-8) - 1

        return image_normalized

