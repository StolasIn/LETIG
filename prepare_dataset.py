import numpy as np
from os import walk
from PIL import Image
from tqdm import tqdm
import torch
from FeatureExtractor.CLIP import CLIP
from FeatureExtractor.VGG import VGG
import json
import configparser


class Attrs:
    def __init__(self):
        pass
    
    def make_dict(self):
        result = dict()
        result['dataset_name'] = self.dataset_name
        result['clip_model'] = self.clip_model
        result['dataset_len'] = self.dataset_len
        result['dimensions'] = self.dimensions
        result['dtype'] = self.dtype
        return result
    
    def built(self, file, dataset_name, cfg):
        self.dataset_name = dataset_name
        self.clip_model = cfg['CLIP']['dis_model']
        self.dataset_len = len(file)
        self.dimensions = len(file[0])
        self.dtype = 'dataset'

def to_json(clip_obj, vgg_obj, name, attrs):
    out_file = open(name, 'w')
    jsonobj = dict()
    jsonobj['attributes'] = attrs.make_dict()

    jsonobj['clip_image_features'] = clip_obj.tolist()
    jsonobj['vgg_image_features'] = vgg_obj.tolist()
    json.dump(jsonobj, out_file, indent=4)

if __name__ == '__main__':
    device = 'cuda:0'
    dataset_name = 'FFHQ'
    config_path = f'configs/prepare_dataset.ini'
    image_path = f'../dataset/{dataset_name}'

    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    filenames = []
    clip_image_features = []
    vgg_image_features = []

    for root, dirs, files in walk(image_path):
        for filename in files:
            filenames.append(filename)
    
    filenames.sort()
    
    clip = CLIP()
    vgg = VGG()
    clip.setup('Default', device)
    vgg.setup('Default', device)
    
    for filename in tqdm(filenames):
        img = Image.open(image_path + "/" + filename)
        with torch.no_grad():
            clip_img_fts = clip.embedding_image(img)
            vgg_img_fts = vgg.embedding(img)
        clip_image_features.append(clip_img_fts.cpu().numpy())
        vgg_image_features.append(vgg_img_fts.cpu().numpy())

    clip_image_features = np.array(clip_image_features)
    vgg_image_features = np.array(vgg_image_features)
    attr = Attrs()
    attr.built(clip_image_features, dataset_name, cfg)
    
    to_json(clip_image_features, vgg_image_features, f'datasets/{dataset_name}-image_features.json', attr)