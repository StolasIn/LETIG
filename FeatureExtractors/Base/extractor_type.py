from enum import Enum


class ExtractorModels(Enum):
    # multimodel
    PE_Core_bigG_14_448 = 1
    ViT_H_14_378_quickgelu = 2
    ViT_gopt_16_SigLIP2_384 = 3
    EVA02_L_14_336 = 4
    ViT_H_14_CLIPA_336 = 5
    convnext_large_d_320 = 6
    coca_ViT_L_14 = 7

    # image
    Inception = 8
    MaxViT = 9
    RegNet = 10
    ResNet = 11
    ResNeXt = 12
    Swin = 13
    VGG = 14
    ViT = 15
    WideResNet = 16