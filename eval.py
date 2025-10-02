from Optimizer import Optimizer
import argparse
import warnings
import configparser
import ImageClass as Img
from EvaluationModel import EvaluatorInfo
from FeatureExtractors import ExtractorModels

warnings.filterwarnings('ignore')

model_similarity_weights = [
        # EvaluatorInfo(model_name = ExtractorModels.PE_Core_bigG_14_448, text_weight = 1.0, image_weight = 0.5, text_dataset_weight=0.0, image_dataset_weight=0.0),
        # EvaluatorInfo(model_name = ExtractorModels.ViT_H_14_378_quickgelu, text_weight = 1.0, image_weight = 0.5, text_dataset_weight=0.0, image_dataset_weight=0.0),
        # EvaluatorInfo(model_name = ExtractorModels.ViT_gopt_16_SigLIP2_384, text_weight = 1.0, image_weight = 0.5, text_dataset_weight=0.0, image_dataset_weight=0.0),
        # EvaluatorInfo(model_name = ExtractorModels.EVA02_L_14_336, text_weight = 1.0, image_weight = 0.5, text_dataset_weight=0.0, image_dataset_weight=0.0),
        # EvaluatorInfo(model_name = ExtractorModels.ViT_H_14_CLIPA_336, text_weight = 1.0, image_weight = 0.5, text_dataset_weight=0.0, image_dataset_weight=0.0),
        # EvaluatorInfo(model_name = ExtractorModels.convnext_large_d_320, text_weight = 1.0, image_weight = 0.5, text_dataset_weight=0.0, image_dataset_weight=0.0),
        # EvaluatorInfo(model_name = ExtractorModels.coca_ViT_L_14, text_weight = 1.0, image_weight = 0.5, text_dataset_weight=0.0, image_dataset_weight=0.0),
        # EvaluatorInfo(model_name = ExtractorModels.Inception, text_weight = 1.0, image_weight = 0.5, text_dataset_weight=0.0, image_dataset_weight=0.0),
        # EvaluatorInfo(model_name = ExtractorModels.MaxViT, text_weight = 1.0, image_weight = 0.5, text_dataset_weight=0.0, image_dataset_weight=0.0),
        # EvaluatorInfo(model_name = ExtractorModels.RegNet, text_weight = 1.0, image_weight = 0.5, text_dataset_weight=0.0, image_dataset_weight=0.0),
        # EvaluatorInfo(model_name = ExtractorModels.ResNet, text_weight = 1.0, image_weight = 0.5, text_dataset_weight=0.0, image_dataset_weight=0.0),
        # EvaluatorInfo(model_name = ExtractorModels.ResNeXt, text_weight = 1.0, image_weight = 0.5, text_dataset_weight=0.0, image_dataset_weight=0.0),
        # EvaluatorInfo(model_name = ExtractorModels.Swin, text_weight = 1.0, image_weight = 0.5, text_dataset_weight=0.0, image_dataset_weight=0.0),
        # EvaluatorInfo(model_name = ExtractorModels.VGG, text_weight = 1.0, image_weight = 0.5, text_dataset_weight=0.0, image_dataset_weight=0.0),
        # EvaluatorInfo(model_name = ExtractorModels.ViT, text_weight = 1.0, image_weight = 0.5, text_dataset_weight=0.0, image_dataset_weight=0.0),
        # EvaluatorInfo(model_name = ExtractorModels.WideResNet, text_weight = 1.0, image_weight = 0.5, text_dataset_weight=0.0, image_dataset_weight=0.0),
    ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--text', type = str)
    parser.add_argument('-g', '--generator', type = str)
    parser.add_argument('-c', '--config', type = str, default = 'configs/mixor10-weight0.5-0.5-s.ini')

    evaluators = [
        EvaluatorInfo(
            model_name = ExtractorModels.ViT_H_14_378_quickgelu, 
            text_weight = 1.0, 
            image_weight = 0.5, 
            text_dataset_weight = 0.0, 
            image_dataset_weight = 0.0, 
            dataset_path = None
        )
    ]
    
    
    args = parser.parse_args()
    
    config = configparser.ConfigParser()
    config.read(args.config)

    model = Optimizer(
        generator_name = args.generator,
        use_txt_feature = False,
        device = config['BASE']['device']
    )

    model.setup_evaluator(
        evaluator_info = evaluators,
        prompt_text = args.text,
        prompt_image = None
    )

    model.setup_optimizer(
        algorithm_name = config['OPTIM']['algorithm'],
        lower_bound = config['OPTIM'].getint('lower_bound'),
        upper_bound = config['OPTIM'].getint('upper_bound'),
        population_size = config['OPTIM'].getint('population_size'),
        n_objectives = config['OPTIM'].getint('n_objectives'),
        n_evaluations = config['OPTIM'].getint('evaluation')
    )

    img, x, f = model.get_fes(verbose=config['BASE'].getboolean('verbose'), seed = config['BASE'].getint('seed'))
    name = args.text.replace(' ', '_').replace('.', '')
    print(f"image name : {name}.png")
    Img.save(img, f'{name}.png')