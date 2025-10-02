import torch
import torch.nn as nn
from FeatureExtractors.Base.base_extractor import BaseExtractor
from torchvision.models import swin_b, Swin_B_Weights

from FeatureExtractors.Base.extractor_type import ExtractorModels

class Swin(nn.Module, BaseExtractor):
    def __init__(self):
        super(Swin, self).__init__()
        self.extractor_flag = "image"
        self.feature_size = 1024

        self.models = {                                             
            ExtractorModels.Swin : "Swin",
        }
    
    def setup(self, model_name, device):
        self.device = device
        self.model_name = model_name
        self.model = self.load_model()

    def load_model(self):
        weights = weights = Swin_B_Weights.DEFAULT
        model = swin_b(weights = weights)
        self.preprocessor = weights.transforms()
        model = model.eval()
        model = model.to(self.device)
        model.fc = nn.Identity()
        return model