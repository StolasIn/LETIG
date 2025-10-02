import torch
import torch.nn as nn
from FeatureExtractors.Base.base_extractor import BaseExtractor
from torchvision.models import inception_v3, Inception_V3_Weights

from FeatureExtractors.Base.extractor_type import ExtractorModels

class Inception(nn.Module, BaseExtractor):
    def __init__(self):
        super(Inception, self).__init__()
        self.extractor_flag = "image"
        self.feature_size = 2048

        self.models = {                                             
            ExtractorModels.Inception : "Inception",
        }
    
    def setup(self, model_name, device):
        self.device = device
        self.model_name = model_name
        self.model = self.load_model()

    def load_model(self):
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights = weights)
        self.preprocessor = weights.transforms()
        model = model.eval()
        model = model.to(self.device)
        model.fc = nn.Identity()
        return model