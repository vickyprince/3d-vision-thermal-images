import torch
import torch.nn as nn
from dust3r.model import AsymmetricCroCo3DStereo
from utils.helpers import get_pointmap

class DUSt3R(nn.Module):
    """
    DUSt3R model for monocular depth estimation built on top of a pretrained AsymmetricCroCo3DStereo model.
    """

    def __init__(self):
        """
        Initialize DUSt3R by loading pretrained weights.
        """
        super().__init__()
        self.model = AsymmetricCroCo3DStereo.from_pretrained("checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth")

    def enable_gradient_checkpointing(self):
        """
        Enable gradient checkpointing in the model to reduce memory usage.
        
        Returns:
            self: The model instance with gradient checkpointing enabled.
        """
        if hasattr(self.model, "enable_gradient_checkpointing"):
            self.model.enable_gradient_checkpointing()
        else:
            if hasattr(self.model, "encoder"):
                self.model.encoder.gradient_checkpointing_enable()
            for module in self.model.modules():
                if hasattr(module, "gradient_checkpointing_enable"):
                    module.gradient_checkpointing_enable()
        return self

    def forward(self, img1, img2):
        """
        Perform a forward pass using two input views.
        
        Args:
            img1 (torch.Tensor): Input tensor for view 1.
            img2 (torch.Tensor): Input tensor for view 2.
            
        Returns:
            dict: A dictionary with keys:
                - 'pointmap1': Extracted pointmap for view 1.
                - 'pointmap2': Extracted pointmap for view 2.
        """
        batch_size = img1.shape[0]
        dummy_instance1 = torch.zeros(batch_size, 1, img1.shape[2], img1.shape[3], device=img1.device)
        dummy_instance2 = torch.zeros(batch_size, 1, img2.shape[2], img2.shape[3], device=img2.device)
        view1 = {"img": img1, "instance": dummy_instance1}
        view2 = {"img": img2, "instance": dummy_instance2}
        
        pred1, pred2 = self.model(view1, view2)
        pm1 = get_pointmap(pred1)
        pm2 = get_pointmap(pred2)
        
        if pm1.dim() == 4 and pm1.shape[1] != 3:
            pm1 = pm1.permute(0, 3, 1, 2)
        if pm2.dim() == 4 and pm2.shape[1] != 3:
            pm2 = pm2.permute(0, 3, 1, 2)
        
        return {'pointmap1': pm1, 'pointmap2': pm2}

    def to(self, device):
        """
        Move the underlying model to the specified device.
        
        Args:
            device: The target device (e.g., 'cuda' or 'cpu').
            
        Returns:
            self: The model moved to the target device.
        """
        self.model = self.model.to(device)
        return self