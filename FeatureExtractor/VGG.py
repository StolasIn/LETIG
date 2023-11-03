import torch
import torchvision.transforms as T

from FeatureExtractor.Base.Extractor import Extractor

class VGG(Extractor):
    def __init__(self):
        pass
    
    def available_models(self):
        models = [
            'vgg11', 'vgg13', 'vgg16', 'vgg19', 'Default'
        ]
        return models
    
    def load_model(self, model_name):
        if model_name == 'Default':
            model_name = 'vgg19'

        model = torch.hub.load("pytorch/vision", model_name, weights="DEFAULT")
        model.classifier = model.classifier[:-1]
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
        