from os import walk
from PIL import Image
from tqdm import tqdm
import torch
import json
from FeatureExtractors import *

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

class Extractors:
    def __init__(self):
        self.models = {
           ExtractorModels.PE_Core_bigG_14_448: ClipOpenExtractor,
           ExtractorModels.ViT_H_14_378_quickgelu: ClipOpenExtractor,
           ExtractorModels.ViT_gopt_16_SigLIP2_384: ClipOpenExtractor,
           ExtractorModels.EVA02_L_14_336: ClipOpenExtractor,
           ExtractorModels.ViT_H_14_CLIPA_336: ClipOpenExtractor,
           ExtractorModels.convnext_large_d_320: ClipOpenExtractor,
           ExtractorModels.coca_ViT_L_14: ClipOpenExtractor,
           ExtractorModels.ViT_SO400M_16_SigLIP2_384: ClipOpenExtractor,
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
    
    def get_model_constructor(self, model: ExtractorModels):
        return self.models[model]

def generate_image_features(model: ExtractorModels, folder_path, batch_size = 5, startpoint = 0):
    def batch(iterable, n = 1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]
    
    def batch_load_images(folder_path, filenames):
        images = []
        for filename in filenames:
            image = Image.open(os.path.join(folder_path, filename))
            images.append(image)
        return images
        
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    E = Extractors()
    M = E.get_model_constructor(model)()
    M.setup(model, device)

    index = []
    filenames = []
    image_features = []

    for root, dirs, files in walk(folder_path):
        for filename in files:
            filenames.append(filename)
    
    filenames.sort()

    if not os.path.exists(f"Features/{M.models[model]}/"):
        os.mkdir(f"Features/{M.models[model]}")

    batch_image_list = list(batch(filenames, n = batch_size))
    batch_image_list = batch_image_list[startpoint:]
    for i, filename in enumerate(tqdm(batch_image_list)):
        images = batch_load_images(folder_path, filename)
        with torch.no_grad(), torch.autocast(device):
            image_features = M.embedding_images(images, batch_size = batch_size)
            index.append(i + startpoint)
            torch.save(image_features, f"Features/{M.models[model]}/{i + startpoint}.pt")
        
    # results = None
    # for i in index:
    #     image_features = torch.load(f"Features/{M.models[model]}/{i}.pt")
    #     if results is None: results = image_features
    #     else: results = torch.cat((results, image_features), dim = 0)

    # torch.save(results, f"Features/{M.models[model]}.pt")

if __name__ == '__main__':
    # model = ExtractorModels.VGG
    # E = Extractors()
    # M = E.get_model_constructor(model)()
    # # M.setup(model, 'cpu')

    # results = None
    # for i in tqdm(range(0, 2800)):
    #     image_features = torch.load(f"Features/{M.models[model]}/{i}.pt")
    #     if results is None: results = image_features
    #     else: results = torch.cat((results, image_features), dim = 0)

    # torch.save(results, f"Features/{M.models[model]}.pt")

    generate_image_features(ExtractorModels.PE_Core_bigG_14_448, "../ImageDatasets/FFHQ/images", batch_size = 5, startpoint = 6752)
    # generate_image_features(ExtractorModels.VGG, "../ImageDatasets/FFHQ/images", batch_size = 25, startpoint = 0)
    # generate_image_features(ExtractorModels.ResNet, "../ImageDatasets/FFHQ/images", batch_size = 25, startpoint = 606)
    # generate_image_features(ExtractorModels.ViT_gopt_16_SigLIP2_384, "../ImageDatasets/FFHQ/images", batch_size = 5, startpoint = 0)

    # device = 'cuda:0'
    # dataset_name = 'FFHQ'
    # config_path = f'configs/prepare_dataset.ini'
    # image_path = f'../dataset/{dataset_name}'

    # cfg = configparser.ConfigParser()
    # cfg.read(config_path)
    # filenames = []
    # clip_image_features = []
    # vgg_image_features = []

    # for root, dirs, files in walk(image_path):
    #     for filename in files:
    #         filenames.append(filename)
    
    # filenames.sort()
    
    # clip = CLIP()
    # vgg = VGG()
    # clip.setup('Default', device)
    # vgg.setup('Default', device)
    
    # for filename in tqdm(filenames):
    #     img = Image.open(image_path + "/" + filename)
    #     with torch.no_grad():
    #         clip_img_fts = clip.embedding_image(img)
    #         vgg_img_fts = vgg.embedding(img)
    #     clip_image_features.append(clip_img_fts.cpu().numpy())
    #     vgg_image_features.append(vgg_img_fts.cpu().numpy())

    # clip_image_features = np.array(clip_image_features)
    # vgg_image_features = np.array(vgg_image_features)
    # attr = Attrs()
    # attr.built(clip_image_features, dataset_name, cfg)
    
    # to_json(clip_image_features, vgg_image_features, f'datasets/{dataset_name}-image_features.json', attr)