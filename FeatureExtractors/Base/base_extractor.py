import torch
from PIL import Image
class BaseExtractor:
    def __init__(
        self
    ):
        self.extractor_flag = "none"
        self.preprocessor = None
    
    def load_image(self, path):
        img = Image.open(path)
        return img
    
    def setup(self, model_name, device):
        self.device = device
        self.model_name = model_name

        self.model = self.load_model(model_name)
        self.model = self.model.to(device)
        self.model.eval()
    
    def load_model(self):
        """
            load pre-trained model to process data
            
            return : model and something related
        """
        
        raise NotImplementedError
    
    # process images
    def preprocess_image(self, image):
        return self.preprocessor(image).unsqueeze(0)
    
    def preprocess_images(self, images):
        return torch.stack([self.preprocessor(image).to(self.device) for image in images])
    
    def embedding_image(self, image):
        image = self.preprocess_image(image)
        with torch.no_grad(), torch.autocast(self.device):
            result = self.model(image)
        return result
    
    def embedding_images(self, images, batch_size = 32):
        preprocess_images = self.preprocess_images(images)
        with torch.no_grad(), torch.autocast(self.device):
            results = None
            batch_images = torch.split(preprocess_images, batch_size)
            for batch_image in batch_images:
                result = self.model(batch_image)
                if results is None: results = result
                else: results = torch.cat((results, result), dim = 0)

        return results

    def feature_similarity(self, fts1, fts2):
        sim = torch.cosine_similarity(fts1, fts2, dim = 0)
        return sim.item()

    def batch_feature_similarity(self, fts1, fts2):
        sim = torch.cosine_similarity(fts1, fts2, dim = 1)
        return sim.float().numpy()