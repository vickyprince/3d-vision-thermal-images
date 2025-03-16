# models/dust3r.py
import torch
import torch.nn as nn
from dust3r.model import AsymmetricCroCo3DStereo
from utils.helpers import get_pointmap 

class DUSt3R(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AsymmetricCroCo3DStereo.from_pretrained("checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth")
    
    def forward(self, img1, img2):
        batch_size = img1.shape[0]
        dummy_instance1 = torch.zeros(batch_size, 1, img1.shape[2], img1.shape[3], device=img1.device)
        dummy_instance2 = torch.zeros(batch_size, 1, img2.shape[2], img2.shape[3], device=img2.device)
        
        view1 = {"img": img1, "instance": dummy_instance1}
        view2 = {"img": img2, "instance": dummy_instance2}
        
        outputs = self.model(view1, view2)
        pred1, pred2 = outputs
        
        # Use helper function to extract the pointmap
        pm1 = get_pointmap(pred1)
        pm2 = get_pointmap(pred2)
        
        # Check if pm1 is channels-last (i.e. shape [B, H, W, 3]) and convert to channels-first.
        if pm1.dim() == 4 and pm1.shape[1] != 3:
            pm1 = pm1.permute(0, 3, 1, 2)
        if pm2.dim() == 4 and pm2.shape[1] != 3:
            pm2 = pm2.permute(0, 3, 1, 2)
        
        return {
            'pointmap1': pm1,
            'pointmap2': pm2
        }
    
    def to(self, device):
        self.model = self.model.to(device)
        return self