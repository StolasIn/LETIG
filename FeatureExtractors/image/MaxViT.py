import torch
import torch.nn as nn
from FeatureExtractors.Base.base_extractor import BaseExtractor
from torchvision.models import maxvit_t, MaxVit_T_Weights

from FeatureExtractors.Base.extractor_type import ExtractorModels

class MaxViT(nn.Module, BaseExtractor):
    def __init__(self):
        super(MaxViT, self).__init__()
        self.extractor_flag = "image"
        self.feature_size = 512

        self.models = {                                             
            ExtractorModels.MaxViT : "MaxViT",
        }
    
    def setup(self, model_name, device):
        self.device = device
        self.model_name = model_name
        self.model = self.load_model()

    def load_model(self):
        weights = MaxVit_T_Weights.DEFAULT
        model = maxvit_t(weights = weights)
        self.preprocessor = weights.transforms()
        model = model.eval()
        model = model.to(self.device)
        model.classifier = model.classifier[:-1]
        return model