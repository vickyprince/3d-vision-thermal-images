import random
import torch
import torchvision.transforms.functional as TF

class ThermalAugmentation:
    """
    Apply basic augmentation to thermal images.

    This class adjusts brightness, contrast, adds Gaussian noise, and applies
    a random horizontal flip to an input image tensor.

    Args:
        brightness (float): Maximum relative brightness adjustment. A factor is randomly chosen from [1 - brightness, 1 + brightness].
        contrast (float): Maximum relative contrast adjustment. A factor is randomly chosen from [1 - contrast, 1 + contrast].
        noise (float): Standard deviation of the additive Gaussian noise.
        flip_prob (float): Probability of applying a horizontal flip.
    """
    def __init__(self, brightness=0.2, contrast=0.2, noise=0.05, flip_prob=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.noise = noise
        self.flip_prob = flip_prob

    def __call__(self, img):
        """
        Apply augmentations to the input image.

        Args:
            img (torch.Tensor): Input image tensor of shape [C, H, W] with values in [0, 1].

        Returns:
            torch.Tensor: Augmented image tensor.
        """
        # Random brightness adjustment
        if self.brightness > 0:
            brightness_factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            img = TF.adjust_brightness(img, brightness_factor)
        
        # Random contrast adjustment
        if self.contrast > 0:
            contrast_factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            img = TF.adjust_contrast(img, contrast_factor)
        
        # Add Gaussian noise
        if self.noise > 0:
            noise_tensor = torch.randn_like(img) * self.noise
            img = img + noise_tensor
            img = torch.clamp(img, 0.0, 1.0)
        
        # Random horizontal flip
        if random.random() < self.flip_prob:
            img = TF.hflip(img)
        
        return img

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create a dummy image tensor (3-channel, 256x256) with random values in [0, 1]
    img = torch.rand(3, 256, 256)
    augmenter = ThermalAugmentation(brightness=0.2, contrast=0.2, noise=0.05, flip_prob=0.5)
    augmented_img = augmenter(img)

    # Visualize the original and augmented images
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img.permute(1, 2, 0).numpy())
    axs[0].set_title("Original")
    axs[0].axis("off")
    axs[1].imshow(augmented_img.permute(1, 2, 0).numpy())
    axs[1].set_title("Augmented")
    axs[1].axis("off")
    plt.tight_layout()
    plt.show()