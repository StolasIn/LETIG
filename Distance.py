import numpy as np
import torch
import json
from FeatureExtractor.CLIP import CLIP
from FeatureExtractor.VGG import VGG

class Datasets:
    
    """
        build feature dataset and evaluation
        
        parameters
        ==========
        
        dataset_path : path to CLIP and VGG feature dataset (str)
        clip_score_threshold : threshold for prepare dataset (float)
        dataset_k : top-k for prepare dataset (int)
        default_batch_size : batch size for evaluation (int, -1 means full batch)
        mode : dataset building method (str, [threshold, topk, normalization, mixand, mixor])
        score_type : type for evaluation (str, [max, mean])
        device : gpu or cpu device (str, [cuda:0, cuda:1])
    """
    
    def __init__(
        self,
        dataset_path,
        clip_score_threshold,
        dataset_k,
        default_batch_size,
        mode,
        score_type,
        device
    ):
        self.device = device
        self.dataset_path = dataset_path
        self.clip_score_threshold = clip_score_threshold
        self.k = dataset_k
        self.mode = mode
        self.score_type = score_type
        self.vgg_weight = 0
        
        self.batch_size = default_batch_size
        if self.batch_size == -1:
            self.batch_size = 1000000
        
        self.dataset_index = dict()
        self.dataset_batch = dict()
        self.clip_image_features = None
        self.vgg_image_features = None
    
    def setup(self, txts, txt_fts):
        self.clip_image_features, self.vgg_image_features = self.load_dataset()
        self.build_dataset(txts, txt_fts)
    
    def load_dataset(self):
        with open(self.dataset_path, newline='') as jsonfile:
            data = json.load(jsonfile)
            clip_image_features = data['clip_image_features']
            vgg_image_features = data['vgg_image_features']
            clip_image_features = np.array(clip_image_features).astype(float)
            vgg_image_features = np.array(vgg_image_features).astype(float)
            clip_image_features = torch.from_numpy(clip_image_features).to(self.device)
            vgg_image_features = torch.from_numpy(vgg_image_features).to(self.device)
        return clip_image_features, vgg_image_features
    
    def get_total_len(self):
        cnt = 0
        for key in self.dataset_index.keys():
            cnt += len(self.dataset_index[key])
        return cnt
    
    def build_dataset(self, txts, txt_fts):
        
        """
            generate index and batch size of a caption
            
            parameters
            ==========
            
            txts : a batch of captions (list, [n])
            txt_fts : a batch of CLIP text features (list, [n, 768])
        """
        
        self.dataset_index = dict()
        self.dataset_batch = dict()
        
        for txt, fts in zip(txts, txt_fts):
            if self.mode == 'threshold':
                indices, batch_size = self.fixed_threshold(fts)
            elif self.mode == 'topk':
                indices, batch_size = self.topk(fts)
            elif self.mode == 'normalization':
                indices, batch_size = self.normalization(fts)
            elif self.mode == 'mixand':
                indices, batch_size = self.mixand(fts)
            elif self.mode == 'mixor':
                indices, batch_size = self.mixor(fts)
            else:
                raise NotImplementedError
            
            self.dataset_index[txt] = indices
            self.dataset_batch[txt] = batch_size
    
    def dataset_similarity(self, txt, clip_img_fts, vgg_img_fts):
        
        """
            calculate image distance for a txt
            
            parameters
            ==========
            
            txt : a caption (str)
            clip_img_fts : a CLIP image feature (tensor, [768])
            vgg_img_fts : a VGG19 image feature (tensor, [4096])
        """
        
        scores = []
        indices = self.dataset_index[txt]
        batch_size = self.dataset_batch[txt]
        batch_index = np.random.choice(indices, batch_size, replace = False)
        for i in batch_index:
            clip_score = self.feature_similarity(self.clip_image_features[i], clip_img_fts)
            
            if self.vgg_weight > 0:
                vgg_score = self.feature_similarity(self.vgg_image_features[i], vgg_img_fts)
                score = clip_score * (1-self.vgg_weight) + vgg_score * self.vgg_weight
            else:
                score = clip_score
                
            scores.append(score)
                
        if len(indices) == 0:
            return 0
        elif self.score_type == 'max':
            return max(scores)
        elif self.score_type == 'mean':
            return sum(scores)/len(scores)
        else:
            raise NotImplementedError
            
    def feature_similarity(self, fts1, fts2):
        return torch.cosine_similarity(fts1, fts2, dim = -1).item()
            
    # process dataset methods -------------------------------------------------------
    def fixed_threshold(self, clip_txt_fts):
        indices = []
        for i in range(len(self.clip_image_features)):
            tmp = self.feature_similarity(self.clip_image_features[i], clip_txt_fts)
            if tmp >= self.clip_score_threshold:
                indices.append(i)

        batch_size = self.batch_size
        batch_size = min(batch_size, len(indices))
        return indices, batch_size

    def topk(self, clip_txt_fts):
        indices = []
        data = []
        for i in range(len(self.clip_image_features)):
            tmp = self.feature_similarity(self.clip_image_features[i], clip_txt_fts)
            data.append([i, tmp])
        
        data.sort(key = lambda x: x[1], reverse=True)
        
        for i in range(self.k):
            indices.append(data[i][0])
        
        batch_size = self.batch_size
        batch_size = min(batch_size, len(indices))
        return indices, batch_size
    
    def normalization(self, clip_txt_fts, sigma = 2.575):
        indices = []
        scores = []
        for i in range(len(self.clip_image_features)):
            tmp = self.feature_similarity(self.clip_image_features[i], clip_txt_fts)
            scores.append(tmp)
        scores = np.array(scores)
        mean = np.mean(scores)
        std = np.std(scores)
        scores = (scores-mean)/std

        for i in range(len(self.clip_image_features)):
            if scores[i] >= sigma:
                indices.append(i)
        
        batch_size = self.batch_size
        batch_size = min(batch_size, len(indices))
        return indices, batch_size
    
    def mixand(self, clip_txt_fts):
        indices, batch_size = self.fixed_threshold(clip_txt_fts)
        if len(indices) < 10:
            indices, batch_size = self.topk(clip_txt_fts)
            
        return indices, batch_size
    
    def mixor(self, clip_txt_fts):
        indices, batch_size = self.fixed_threshold(clip_txt_fts)
        if len(indices) == 0:
            indices, batch_size = self.topk(clip_txt_fts)
            
        return indices, batch_size    
        
class Distance:
    
    """
        calcualte image distance and semantic distance of an image
        
        parameters
        ==========
        
        dataset_path : path to CLIP and VGG feature dataset (str)
        clip_score_threshold : threshold for prepare dataset (float)
        dataset_k : top-k for prepare dataset (int)
        default_batch_size : batch size for evaluation (int, -1 means full batch)
        mode : dataset building method (str, [threshold, topk, normalization, mixand, mixor])
        score_type : type for evaluation (str, [max, mean])
        device : gpu or cpu device (str, [cuda:0, cuda:1])
    """
    
    def __init__(
        self,
        dataset_path,
        clip_score_threshold,
        dataset_k,
        default_batch_size,
        mode,
        score_type,
        device
    ):
        self.device = device
        self.vgg_weight = 0
        
        self.clip = CLIP()
        if self.vgg_weight > 0:
            self.vgg = VGG()
        
        self.Dataset = Datasets(dataset_path, clip_score_threshold, dataset_k, default_batch_size, mode, score_type, device)
        self.txt_fts = dict()
        
    def setup(self, txts):
        self.clip.setup("Default", self.device)
        
        if self.vgg_weight > 0:
            self.vgg.setup("Default", self.device)

        self.txt_fts = dict()
        for txt in txts:
            self.txt_fts[txt] = self.clip.embedding_text(txt)
            
        self.Dataset.setup(txts, list(self.txt_fts.values()))
        
    def get_dataset_len(self):
        return self.Dataset.get_total_len()
    
    # calculate semantic score and realistic score
    def distance_metric(self, txt, img):
        
        # embedding text and image
        clip_txt_fts = self.txt_fts[txt]
        clip_img_fts = self.clip.embedding_image(img)
        
        if self.vgg_weight > 0:
            vgg_img_fts = self.vgg.embedding(img)
        else:
            vgg_img_fts = None
        
        semantic_similarity = self.semantic_similarity(clip_txt_fts, clip_img_fts)
        image_similarity = self.image_similarity(txt, clip_img_fts, vgg_img_fts)
        
        return semantic_similarity, image_similarity
    
    def image_similarity(self, txt, clip_img_fts, vgg_img_fts):
        return self.Dataset.dataset_similarity(txt, clip_img_fts, vgg_img_fts)
        
    def semantic_similarity(self, clip_txt_fts, clip_img_fts):
        return self.clip.feature_similarity(clip_txt_fts, clip_img_fts)
