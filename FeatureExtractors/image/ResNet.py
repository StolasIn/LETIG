import torch
import torch.nn as nn
from FeatureExtractors.Base.base_extractor import BaseExtractor
from torchvision.models import resnet152, ResNet152_Weights

from FeatureExtractors.Base.extractor_type import ExtractorModels

class ResNet(nn.Module, BaseExtractor):
    def __init__(self):
        super(ResNet, self).__init__()
        self.extractor_flag = "image"
        self.feature_size = 2048

        self.models = {                                             
            ExtractorModels.ResNet : "ResNet",
        }
    
    def setup(self, model_name, device):
        self.device = device
        self.model_name = model_name
        self.model = self.load_model()

    def load_model(self):
        weights = ResNet152_Weights.DEFAULT
        model = resnet152(weights = weights)
        self.preprocessor = weights.transforms()
        model = model.eval()
        model = model.to(self.device)
        model.fc = nn.Identity()
        return model