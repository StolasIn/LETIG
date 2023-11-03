import torch
import torch.nn as nn
import torchvision.transforms as T

from Base.Extractor import Extractor

class ResNet(Extractor):
    def __init__(self):
        pass
    
    def available_models(self):
        models = [
            'resner18', 'resner34', 'resner50', 'resner101', 'resnet152', 'Default'
        ]
        return models
    
    def load_model(self, model_name):
        if model_name == 'Default':
            model_name = 'resnet152'

        model = torch.hub.load("pytorch/vision", model_name, weights="DEFAULT")
        modules = list(model.children())[:-1]
        model = nn.Sequential(*modules)
        for p in model.parameters():
            p.requires_grad = False
        return model
    
    def preprocess(self, img):
        
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        img = transform(img)
        return img
    
    def embedding(self, img):
        preprocess_imgs = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(preprocess_imgs)
        return output.flatten()
    
    def similarity(self, img1, img2):
        embedding1 = self.embedding(img1)
        embedding2 = self.embedding(img2)
        
        return self.feature_similarity(embedding1, embedding2)
        