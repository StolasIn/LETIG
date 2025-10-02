import torch
import torch.nn as nn
from FeatureExtractors.Base.base_extractor import BaseExtractor
from torchvision.models import regnet_x_16gf, RegNet_X_16GF_Weights

from FeatureExtractors.Base.extractor_type import ExtractorModels

class RegNet(nn.Module, BaseExtractor):
    def __init__(self):
        super(RegNet, self).__init__()
        self.extractor_flag = "image"
        self.feature_size = 2048

        self.models = {                                             
            ExtractorModels.RegNet : "RegNet",
        }
    
    def setup(self, model_name, device):
        self.device = device
        self.model_name = model_name
        self.model = self.load_model()

    def load_model(self):
        weights = RegNet_X_16GF_Weights.DEFAULT
        model = regnet_x_16gf(weights = weights)
        self.preprocessor = weights.transforms()
        model = model.eval()
        model = model.to(self.device)
        model.fc = nn.Identity()
        return model