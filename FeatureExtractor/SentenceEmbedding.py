import torch
from sentence_transformers import SentenceTransformer

from FeatureExtractor.Base.Extractor import Extractor

class SentenceEmbedding(Extractor):
    def __init__(self):
        pass
    
    def available_models(self):
        models = ['all-mpnet-base-v2', 'Default']

        raise models 
    
    def load_model(self, model_name):
        if model_name == 'Default':
            model_name = 'all-mpnet-base-v2'
        
        model = SentenceTransformer(model_name)

        return model

    def embedding(self, txt):
        with torch.no_grad():
            embeddings = self.model.encode([txt])
        return embeddings[0]
    
    
    def similarity(self, txt1, txt2):
        embedding1 = self.embedding(txt1)
        embedding2 = self.embedding(txt2)

        return self.feature_similarity(embedding1, embedding2)
        

    