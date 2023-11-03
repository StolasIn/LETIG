import torch
from Generator import Generator
from Distance import Distance
import Optimizer

class Util:
    def __init__(self, generator_name, dataset_name, config):
        torch.autograd.set_grad_enabled(False)
        self.config = config
        self.device = config['BASE']['device']
        self.txts = None
        
        # Generator
        self.generator_path = f'checkpoints/{generator_name}'
        self.use_fts = config['BASE'].getboolean('use_txt_feature')
        
        # Distance
        self.dataset_path = f'datasets/{dataset_name}-image_features.json'
        self.clip_score_threshold = config['DISTANCE'].getfloat('preprocess_threshold')
        self.default_batch_size = config['DISTANCE'].getint('batch_size')
        self.mode = config['DISTANCE']['mode']
        self.score_type = config['DISTANCE']['score_type']
        self.dataset_k = config['DISTANCE'].getint('k')
        
        # classes
        self.generator = Generator(self.generator_path, self.use_fts, self.device)
        self.distance = Distance(self.dataset_path, self.clip_score_threshold, self.dataset_k, self.default_batch_size, self.mode, self.score_type, self.device)
    
    def setup(self, txts):
        self.distance.setup(txts)
        self.txts = txts
    
    def get_dataset_len(self):
        return self.distance.get_dataset_len()

    def get_score(self, txt, img):
        semantic_score, realistic_score = self.distance.distance_metric(txt, img)
        semantic_score *= 100
        realistic_score *= 100

        return [semantic_score, realistic_score]

    def get_scores(self, txt, ws):
        semantic_scores = []
        realistic_scores = []
        for w in ws:
            img = self.generator.get_img_from_w(w = w)
            scores = self.get_score(txt = txt, img = img)
            semantic_score, realistic_score = scores[0], scores[1]
            
            # -1 for maximization
            semantic_scores.append(-semantic_score)
            realistic_scores.append(-realistic_score)

        return [semantic_scores, realistic_scores]
        
    def get_fes(self):
        images = []
        xs, fs = Optimizer.solve_multitask(self, self.txts, self.config)
        for x in xs:
            w = torch.from_numpy(x).to(self.device)
            images.append(self.generator.get_img_from_w(w = w))
        return images, xs, fs