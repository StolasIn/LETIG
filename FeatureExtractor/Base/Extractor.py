import torch
from PIL import Image
class Extractor:
    def __init__(
        self
    ):
        pass
    
    def load_image(self, path):
        img = Image.open(path)
        return img
    
    def setup(self, model_name, device):
        self.device = device
        self.model_name = model_name

        self.model = self.load_model(model_name)
        self.model = self.model.to(device)
        self.model.eval()
    
    def available_models(self):
        """
            available models
            
            return : available models name
        """

        raise NotImplementedError 
    
    def load_model(self):
        """
            load pre-trained model to process data
            
            return : model and something related
        """
        
        raise NotImplementedError
    
    def preprocess(self):
        """
            propresess data
            
            return : a propresessed data
        """
        
        raise NotImplementedError
    
    def embedding(self):
        """
            generate embedding according given data
            
            return : data embeddings
        """
        
        raise NotImplementedError
    
    def similarity(self):
        """
            given two data, predict similarity between then
            
            return : a score measure how similar the two example is
        """
        
        raise NotImplementedError

    def feature_similarity(self, fts1, fts2):
        sim = torch.cosine_similarity(fts1, fts2, dim = 0)
        return sim.item()