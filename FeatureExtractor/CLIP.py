import torch
import clip

from FeatureExtractor.Base.Extractor import Extractor

class CLIP(Extractor):
    def __init__(self):
        pass
    
    def available_models(self):
        models = [
            'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px', 'Default'
        ]
        return models
    
    def setup(self, model_name, device, store_txt_fts = True):
        self.device = device
        self.model_name = model_name
        self.store_txt_fts = store_txt_fts
        self.txt_fts = {}

        self.model, self.clip_preprocess = self.load_model(model_name)
        self.model = self.model.to(device)
        self.model.eval()
    
    def load_model(self, model_name):
        if model_name == 'Default':
            model_name = 'ViT-L/14@336px'

        model, clip_preprocess = clip.load(model_name, device=self.device, jit=False)
        model = model.eval()
        model = model.to(self.device)
        return model, clip_preprocess
    
    def preprocess_image(self, image):

        """
            preprocess image to proper resolution by scale (bicubic interpolation)
            334x334 for clip model

            *unsqueeze to generate batch for single image input [1, 3, 334, 334]
        """

        return self.clip_preprocess(image).unsqueeze(0).to(self.device)

    def preprocess_text(self, txt):
        return clip.tokenize([txt]).to(self.device)

    def embedding_text(self, txt):

        """
            tokenize text and encoding with discriminative clip model

            *normalize to unit vector
            *store text features if memory is available
        """

        if txt in self.txt_fts:
            return self.txt_fts[txt]

        preprocess_txt = self.preprocess_text(txt)
        txt_fts = self.model.encode_text(preprocess_txt)
        txt_fts = txt_fts/txt_fts.norm(dim=-1, keepdim=True)
        if self.store_txt_fts == True:
            self.txt_fts[txt] = txt_fts[0]

        return txt_fts[0]
    
    def embedding_image(self, img):

        """
            encoding image with discriminative clip model

            *normalize to unit vector
        """

        preprocess_img = self.preprocess_image(img)
        img_fts = self.model.encode_image(preprocess_img)
        img_fts /= img_fts.norm(dim=-1, keepdim=True)
        return img_fts[0]
    
    def similarity(self, txt, img):

        """ 
            calculate cosine similarity between text and image embeddings

            *item() to detech
        """

        txt_fts = self.embedding_text(txt = txt)
        img_fts = self.embedding_image(img = img)
        return self.feature_similarity(txt_fts, img_fts)
    
    def match_image_with_text(self, img, txts):
        img_fts = self.embedding_image(img)
        ma = -1
        result = 0
        for i in range(len(txts)):
            txt_fts = self.embedding_text(txts[i])
            sim = self.feature_similarity(txt_fts, img_fts)
            if sim > ma :
                ma = sim
                result = i

        return txts[result]
    
    def match_text_with_image(self, txt, imgs):
        txt_fts = self.embedding_text(txt)
        ma = -1
        result = 0
        for i in range(len(imgs)):
            img_fts = self.embedding_image(imgs[i])
            sim = self.feature_similarity(txt_fts, img_fts)
            if sim > ma :
                ma = sim
                result = i

        return imgs[result]