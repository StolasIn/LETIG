import numpy as np
from FeatureExtractors import *

class EvaluatorInfo:
    def __init__(
        self, 
        model_name, 
        text_weight, 
        image_weight,
        text_dataset_weight,
        image_dataset_weight,
        dataset_path = None
    ):
        self.model_name = model_name
        self.text_weight = text_weight
        self.image_weight = image_weight
        self.text_dataset_weight = text_dataset_weight
        self.image_dataset_weight = image_dataset_weight
        self.dataset_path = dataset_path

def canEmbedText(extractor: BaseExtractor) -> bool:
    return extractor.extractor_flag == "multimodel" or extractor.extractor_flag == "text"

def canEmbedImage(extractor: BaseExtractor) -> bool:
    return extractor.extractor_flag == "multimodel" or extractor.extractor_flag == "image"

class EvaluationModel:
    def __init__(self):
        self.models = {
           ExtractorModels.PE_Core_bigG_14_448: ClipOpenExtractor,
           ExtractorModels.ViT_H_14_378_quickgelu: ClipOpenExtractor,
           ExtractorModels.ViT_gopt_16_SigLIP2_384: ClipOpenExtractor,
           ExtractorModels.EVA02_L_14_336: ClipOpenExtractor,
           ExtractorModels.ViT_H_14_CLIPA_336: ClipOpenExtractor,
           ExtractorModels.convnext_large_d_320: ClipOpenExtractor,
           ExtractorModels.coca_ViT_L_14: ClipOpenExtractor,
           ExtractorModels.Inception: Inception,
           ExtractorModels.MaxViT: MaxViT,
           ExtractorModels.RegNet: RegNet,
           ExtractorModels.ResNet: ResNet,
           ExtractorModels.ResNeXt: ResNeXt,
           ExtractorModels.Swin: Swin,
           ExtractorModels.VGG: VGG,
           ExtractorModels.ViT: ViT,
           ExtractorModels.WideResNet: WideResNet,
        }
    
    def setup(
        self, 
        prompt_text, 
        prompt_image, 
        evaluator_info, 
        n_dataset_samples, 
        dataset_threshold,
        device
    ):
        self.prompt_text = prompt_text
        self.prompt_image = prompt_image
        self.prompt_text_features = dict()
        self.prompt_image_features = dict()

        self.evaluator_info = evaluator_info
        self.device = device
        self.extractor: list[BaseExtractor] = []
        self.text_similarity_datasets = dict()
        self.image_similarity_datasets = dict()
        for i in range(len(self.evaluator_info)):
            self.extractor.append(self.models[self.evaluator_info[i].model_name]())
        self.prebuild_prompt_features(n_dataset_samples, dataset_threshold)

    def prebuild_prompt_features(self, n_dataset_samples, dataset_threshold):
        for i, extractor in enumerate(self.extractor):
            extractor.setup(self.evaluator_info[i].model_name, self.device)

            if self.evaluator_info[i].dataset_path != None:
                features = torch.load(self.evaluator_info[i].dataset_path)
                features = features.to(self.device)
            else:
                features = None

            self.text_similarity_datasets[extractor.models[extractor.model_name]] = None
            self.image_similarity_datasets[extractor.models[extractor.model_name]] = None
            self.prompt_text_features[extractor.models[extractor.model_name]] = None
            self.prompt_image_features[extractor.models[extractor.model_name]] = None

            if canEmbedText(extractor) and self.prompt_text is not None:
                self.prompt_text_features[extractor.models[extractor.model_name]] = extractor.embedding_text(self.prompt_text)

                if features is not None and self.evaluator_info[i].text_dataset_weight > 0:
                    similaritys = extractor.batch_feature_similarity(self.prompt_text_features[extractor.models[extractor.model_name]], features)
                    self.text_similarity_datasets[extractor.models[extractor.model_name]] = self.build_dataset(features, similaritys, n_dataset_samples, dataset_threshold)

            if canEmbedImage(extractor) and self.prompt_image is not None:
                self.prompt_image_features[extractor.models[extractor.model_name]] = extractor.embedding_image(self.prompt_image)

                if features is not None and self.evaluator_info[i].image_dataset_weight > 0:
                    similaritys = extractor.batch_feature_similarity(self.prompt_image_features[extractor.models[extractor.model_name]], features)
                    self.image_similarity_datasets[extractor.models[extractor.model_name]] = self.build_dataset(features, similaritys, n_dataset_samples, dataset_threshold)
            
            del extractor

    def build_dataset(self, features, similaritys, n_dataset_samples, dataset_threshold):
        if features is None:
            return None
        features = features[similaritys >= dataset_threshold]
        similaritys = similaritys[similaritys >= dataset_threshold]
        index = np.argsort(-similaritys)[:n_dataset_samples]
        return features[index]

    def get_text_similaritys(self, extractor: BaseExtractor, weight, text_feature, generated_image_features):
        if weight <= 0 or not canEmbedText(extractor) or self.prompt_text is None or text_feature is None:
            return np.zeros((len(generated_image_features)))
        
        return extractor.batch_feature_similarity(self.prompt_text_features[extractor.models[extractor.model_name]], generated_image_features)
    
    def get_image_similaritys(self, extractor: BaseExtractor, weight, image_feature, generated_image_features):
        if weight <= 0 or not canEmbedImage(extractor) or self.prompt_image is None or image_feature is None:
            return np.zeros((len(generated_image_features)))

        return extractor.batch_feature_similarity(self.prompt_image_features[extractor.models[extractor.model_name]], generated_image_features)

    def get_dataset_similaritys(self, extractor: BaseExtractor, weight, image_features, generated_image_features):
        if weight <= 0 or image_features is None:
            return np.zeros((len(generated_image_features)))
        
        similaritys = np.zeros((len(generated_image_features)))
        for i in range(len(generated_image_features)):
            similarity = extractor.batch_feature_similarity(image_features, generated_image_features)
            similarity = np.mean(similarity, axis = 0)
            similaritys[i] = similarity
        return similaritys

    def calculate_similaritys(self, generated_images, batch_size = 8):
        results = np.zeros((len(generated_images), 4))

        for i, extractor in enumerate(self.extractor):
            extractor.setup(self.evaluator_info[i].model_name, self.device)
            generated_image_features = extractor.embedding_images(generated_images, batch_size = batch_size)

            text_similaritys = self.get_text_similaritys(
                extractor, 
                self.evaluator_info[i].text_weight, 
                self.prompt_text_features[extractor.models[extractor.model_name]], 
                generated_image_features
            )
            
            image_similaritys = self.get_image_similaritys(
                extractor, 
                self.evaluator_info[i].image_weight, 
                self.prompt_image_features[extractor.models[extractor.model_name]], 
                generated_image_features
            )

            text_dataset_similarity = self.get_dataset_similaritys(
                extractor, 
                self.evaluator_info[i].text_dataset_weight, 
                self.text_similarity_datasets[extractor.models[extractor.model_name]], 
                generated_image_features
            )

            image_dataset_similarity = self.get_dataset_similaritys(
                extractor, 
                self.evaluator_info[i].image_dataset_weight, 
                self.image_similarity_datasets[extractor.models[extractor.model_name]], 
                generated_image_features
            )

            results[:, 0] += self.evaluator_info[i].text_weight * text_similaritys
            results[:, 1] += self.evaluator_info[i].image_weight * image_similaritys
            results[:, 2] += self.evaluator_info[i].text_dataset_weight * text_dataset_similarity
            results[:, 3] += self.evaluator_info[i].image_dataset_weight * image_dataset_similarity

            print(results)
            del extractor

        results /= len(self.extractor)
        return results