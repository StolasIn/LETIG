import torch
import torch.nn as nn
from tqdm import tqdm
from FeatureExtractors.Base.base_extractor import BaseExtractor
from torchvision.models import vgg19, VGG19_Weights

from FeatureExtractors.Base.extractor_type import ExtractorModels

class VGG(nn.Module, BaseExtractor):
    def __init__(self):
        super(VGG, self).__init__()
        self.extractor_flag = "image"
        self.feature_size = 4096

        self.models = {                                             
            ExtractorModels.VGG : "VGG",
        }
    
    def setup(self, model_name, device):
        self.device = device
        self.model_name = model_name
        self.model = self.load_model()

    def load_model(self):
        weights = VGG19_Weights.DEFAULT
        model = vgg19(weights = weights)
        self.preprocessor = weights.transforms()
        model = model.eval()
        model = model.to(self.device)
        model.classifier = model.classifier[:-1]
        return model