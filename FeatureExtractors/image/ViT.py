import torch
import torch.nn as nn
from FeatureExtractors.Base.base_extractor import BaseExtractor
from torchvision.models import vit_l_16, ViT_L_16_Weights

from FeatureExtractors.Base.extractor_type import ExtractorModels

class ViT(nn.Module, BaseExtractor):
    def __init__(self):
        super(ViT, self).__init__()
        self.extractor_flag = "image"
        self.feature_size = 1024
        
        self.models = {                                             
            ExtractorModels.ViT : "ViT",
        }
    
    def setup(self, model_name, device):
        self.device = device
        self.model_name = model_name
        self.model = self.load_model()

    def load_model(self):
        weights = ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
        self.preprocessor = weights.transforms()
        model = vit_l_16(weights = weights)
        model = model.eval()
        model = model.to(self.device)
        model.fc = nn.Identity()
        return model