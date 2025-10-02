import torch
import torch.nn as nn
import torch.nn as nn
from FeatureExtractors.Base.base_extractor import BaseExtractor
from torchvision.models import wide_resnet101_2, Wide_ResNet101_2_Weights

from FeatureExtractors.Base.extractor_type import ExtractorModels

class WideResNet(nn.Module, BaseExtractor):
    def __init__(self):
        super(WideResNet, self).__init__()
        self.extractor_flag = "image"
        self.feature_size = 2048

        self.models = {                                             
            ExtractorModels.WideResNet : "WideResNet",
        }
    
    def setup(self, model_name, device):
        self.device = device
        self.model_name = model_name
        self.model = self.load_model()

    def load_model(self):
        weights = Wide_ResNet101_2_Weights.DEFAULT
        model = wide_resnet101_2(weights = weights)
        self.preprocessor = weights.transforms()
        model = model.eval()
        model = model.to(self.device)
        model.fc = nn.Identity()
        return model