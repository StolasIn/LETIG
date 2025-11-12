import open_clip
import torch
from tqdm import tqdm

from FeatureExtractors.Base.base_extractor import BaseExtractor
from FeatureExtractors.Base.extractor_type import ExtractorModels

class ClipOpenExtractor(BaseExtractor):
    def __init__(self):
        self.extractor_flag = "multimodel"
        self.pretrain = {                                                         #  acc    params  nflops
            ExtractorModels.PE_Core_bigG_14_448 : "meta",                         # ----------------------
            ExtractorModels.ViT_H_14_378_quickgelu : "dfn5b",                     # 0.7079, 986.71, 1054.0
            ExtractorModels.ViT_gopt_16_SigLIP2_384 : "webli",                    # 0.6921, 877.96, 723.48
            ExtractorModels.EVA02_L_14_336 : "merged2b_s6b_b61k",                 # 0.6583, 428.08, 395.16
            ExtractorModels.ViT_H_14_CLIPA_336 : "datacomp1b",                    # 0.6439, 968.64, 800.88
            ExtractorModels.convnext_large_d_320 : "laion2b_s29b_b131k_ft_soup",  # 0.6387, 351.77, 157.98
            ExtractorModels.coca_ViT_L_14 : "mscoco_finetuned_laion2b_s13b_b90k", # 0.6327, 638.45, 214.52
            ExtractorModels.ViT_SO400M_16_SigLIP2_384 : "webli",                  
        }

        self.models = {                                             
            ExtractorModels.PE_Core_bigG_14_448 : "PE-Core-bigG-14-448",
            ExtractorModels.ViT_H_14_378_quickgelu : "ViT-H-14-378-quickgelu",
            ExtractorModels.ViT_gopt_16_SigLIP2_384 : "ViT-gopt-16-SigLIP2-384",
            ExtractorModels.EVA02_L_14_336 : "EVA02-L-14-336",
            ExtractorModels.ViT_H_14_CLIPA_336 : "ViT-H-14-CLIPA-336",
            ExtractorModels.convnext_large_d_320 : "convnext_large_d_320",
            ExtractorModels.coca_ViT_L_14 : "coca_ViT-L-14",
            ExtractorModels.ViT_SO400M_16_SigLIP2_384 : "ViT-SO400M-16-SigLIP2-384",
        }
    
    def available_models(self):
        return self.models
    
    def setup(self, model_name, device = "cpu", store_text_features = True):
        self.device = device
        self.model_name = model_name
        self.store_text_features = store_text_features
        self.text_features = {}

        self.model, self.text_tokenizer, self.image_preprocessor = self.load_model(model_name)
        self.model = self.model.to(device)
        self.model.eval()
    
    def load_model(self, model_name):
        model, _, image_preprocessor = open_clip.create_model_and_transforms(self.models[model_name], pretrained=self.pretrain[model_name])
        text_tokenizer = open_clip.get_tokenizer(self.models[model_name])
        model = model.eval()
        model = model.to(self.device)
        return model, text_tokenizer, image_preprocessor
    
    def preprocess_image(self, image):
        return self.image_preprocessor(image).unsqueeze(0).to(self.device)

    def preprocess_text(self, text):
        return self.text_tokenizer([text]).to(self.device)
    
    def preprocess_images(self, images):
        return torch.stack([self.image_preprocessor(image).to(self.device) for image in images])

    def preprocess_texts(self, texts):
        return self.text_tokenizer(texts).to(self.device)

    def embedding_text(self, text):

        """
            tokenize text and encoding with discriminative clip model

            *normalize to unit vector
            *store text features if memory is available
        """

        if text in self.text_features:
            return self.text_features[text]

        with torch.no_grad(), torch.autocast(self.device):
            preprocess_text = self.preprocess_text(text)
            text_features = self.model.encode_text(preprocess_text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            if self.store_text_features == True:
                self.text_features[text] = text_features[0]

        return text_features[0]
    
    def embedding_image(self, image):

        """
            encoding image with discriminative clip model

            *normalize to unit vector
        """
        with torch.no_grad(), torch.autocast(self.device):
            preprocess_image = self.preprocess_image(image)
            image_features = self.model.encode_image(preprocess_image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features[0]
    
    def embedding_texts(self, texts):

        """
            tokenize text and encoding with discriminative clip model

            *normalize to unit vector
            *store text features if memory is available
        """

        with torch.no_grad(), torch.autocast(self.device):
            preprocess_texts = self.preprocess_texts(texts)
            text_features = self.model.encode_text(preprocess_texts)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features
    
    def embedding_images(self, images, batch_size = 32):

        """
            encoding image with discriminative clip model

            *normalize to unit vector
        """
        with torch.no_grad(), torch.autocast(self.device):
            preprocess_images = self.preprocess_images(images)
            batch_images = torch.split(preprocess_images, batch_size)
            results = None

            for batch_image in batch_images:
                image_features = self.model.encode_image(batch_image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                if results is None: results = image_features
                else: results = torch.cat((results, image_features), dim = 0)

        return results
    
    def similarity(self, text, image):

        """ 
            calculate cosine similarity between text and image embeddings

            *item() to detech
        """

        text_features = self.embedding_text(text = text)
        image_features = self.embedding_image(image = image)
        return self.feature_similarity(text_features, image_features)
    
    def image_similarity(self, image1, image2):
        image_features1 = self.embedding_image(image = image1)
        image_features2 = self.embedding_image(image = image2)
        return self.feature_similarity(image_features1, image_features2)
    
    def match_image_with_text(self, image, texts):
        image_features = self.embedding_image(image)
        ma = -1
        result = 0
        for i in range(len(texts)):
            text_features = self.embedding_text(texts[i])
            sim = self.feature_similarity(text_features, image_features)
            if sim > ma :
                ma = sim
                result = i

        return texts[result]
    
    def match_text_with_image(self, text, images):
        text_features = self.embedding_text(text)
        ma = -1
        result = 0
        for i in range(len(images)):
            image_features = self.embedding_image(images[i])
            sim = self.feature_similarity(text_features, image_features)
            if sim > ma :
                ma = sim
                result = i

        return images[result]
    