import os
import copy
import abc
import torch
import torchvision

from tds_sim.models.tools.quanitzation import QuantizationModule


class _MetaModelConfig(metaclass=abc.ABCMeta):
    def __init__(self, model_type, weights=None):
        self.model_type = model_type
        self.weights = weights

    @abc.abstractmethod
    def generate(self):
        pass

class ModelConfig(_MetaModelConfig):
    def __init__(self, model_type, weights=None):
        super(ModelConfig, self).__init__(model_type, weights)

    def generate(self) -> torch.nn.Module:
        return self.model_type(weights=self.weights)

class QuantModelConfig(_MetaModelConfig):
    def __init__(self, model_type, weights=None):
        super(QuantModelConfig, self).__init__(model_type, weights)

    def generate(self):
        return self.model_type(weights=self.weights, quantize=True)


def generate_from_chkpoint(model_primitive: torch.nn.Module, chkpoint_path: str):
    state_dict = torch.load(chkpoint_path)
    model = copy.deepcopy(model_primitive)
    model.load_state_dict(state_dict)
    return model

def generate_from_quant_chkpoint(model_primitive: torch.nn.Module, chkpoint_path: str, example_inputs):
    state_dict = torch.load(chkpoint_path)
    qmod = QuantizationModule()
    qmodel = qmod.quantize(model_primitive, example_inputs, citer=0, verbose=0)
    qmodel.load_state_dict(state_dict)
    return qmodel

def generate_quant_from_primitive(model_primitive: torch.nn.Module, example_inputs):
    qmod = QuantizationModule()
    qmodel = qmod.quantize(model_primitive, example_inputs, citer=0, verbose=0, default_qconfig='x86')
    return qmodel


imagenet_pretrained: dict[str, ModelConfig] = {
    'ResNet50': ModelConfig(
        torchvision.models.resnet50,
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1,
    ),
    'ResNet34': ModelConfig(
        torchvision.models.resnet34,
        weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1,
    ),
    'ResNet18': ModelConfig(
        torchvision.models.resnet18,
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
    ),
    'AlexNet': ModelConfig(
        torchvision.models.alexnet,
        weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1.IMAGENET1K_V1,
    ),
    'VGG16': ModelConfig(
        torchvision.models.vgg16,
        weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1,
    ),
    'SqueezeNet': ModelConfig(
        torchvision.models.squeezenet1_0,
        weights=torchvision.models.SqueezeNet1_0_Weights.IMAGENET1K_V1,
    ),
    'InceptionV3': ModelConfig(
        torchvision.models.inception_v3,
        weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1,
    ),
    'EfficientNetb0': ModelConfig(
        torchvision.models.efficientnet_b0,
        weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    ),
    'MNASNet13': ModelConfig(
        torchvision.models.mnasnet1_3,
        weights=torchvision.models.MNASNet1_3_Weights.IMAGENET1K_V1,
    ),
    'GoogLeNet': ModelConfig(
        torchvision.models.googlenet,
        weights=torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1,
    ),
    'MobileNetV2': ModelConfig(
        torchvision.models.mobilenet_v2,
        weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1,
    ),
}

imagenet_quant_pretrained = {
    'ResNet50': QuantModelConfig(
        torchvision.models.quantization.resnet50,
        weights=torchvision.models.quantization.ResNet50_QuantizedWeights.IMAGENET1K_FBGEMM_V1,
    ),
    'GoogLeNet': QuantModelConfig(
        torchvision.models.quantization.googlenet,
        weights=torchvision.models.quantization.GoogLeNet_QuantizedWeights.IMAGENET1K_FBGEMM_V1,
    ),
    'InceptionV3': QuantModelConfig(
        torchvision.models.quantization.inception_v3,
        weights=torchvision.models.quantization.Inception_V3_QuantizedWeights.IMAGENET1K_FBGEMM_V1,
    ),
}