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
    ViT_SO400M_16_SigLIP2_384 = 8

    # image
    Inception = 9
    MaxViT = 10
    RegNet = 11
    ResNet = 12
    ResNeXt = 13
    Swin = 14
    VGG = 15
    ViT = 16
    WideResNet = 17