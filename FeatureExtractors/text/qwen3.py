import os
from sentence_transformers import SentenceTransformer
from FeatureExtractors.Base.base_extractor import BaseExtractor

class Qwen3Extractor(BaseExtractor):
    def __init__(self):
        self.extractor_flag = "text"

    def setup(self, model_name, device):
        if model_name == "Default":
            model_name = "Qwen/Qwen3-Embedding-0.6B"
            
        self.device = device
        self.model_name = model_name
        self.model =  SentenceTransformer(model_name)

    def embedding_text(self, text, prompt_name = None):
        if prompt_name == None:
            return self.model.encode(text, convert_to_numpy = False)
        return self.model.encode(text, prompt_name = prompt_name, convert_to_numpy = False)

    def similarity(self, text1, text2):
        text_features1 = self.embedding_text(text = text1)
        text_features2 = self.embedding_text(text = text2)
        return self.feature_similarity(text_features1, text_features2)