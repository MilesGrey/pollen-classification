import torch
from torchvision.transforms.functional import pad


class SquarePad:
    def __call__(self, image):
        height, width = image.size()[-2:]
        max_side = max([width, height])
        horizontal_pad = int((max_side - width) / 2)
        vertical_pad = int((max_side - height) / 2)
        padded_image = pad(image, [horizontal_pad, vertical_pad, horizontal_pad, vertical_pad], 0, 'constant')
        return padded_image


class AdditiveWhiteGaussianNoise:
    def __init__(self, mean, standard_deviation):
        self.mean = mean
        self.standard_deviation = standard_deviation

    def __call__(self, image):
        return image + (self.mean + self.standard_deviation * torch.randn_like(image))
