from __future__ import print_function
import os
import numpy as np
import random
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageFilter, ImageEnhance


def randomFlip(image, prob=0.5):
    rnd = np.random.random_sample()
    if rnd < prob:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    return image

def randomBlur(image, prob=0.5):
    rnd = np.random.random_sample()
    if rnd < prob:
        return image.filter(ImageFilter.BLUR)
    return image

def randomRotation(image, prob=0.5, angle=(1, 60)):
    rnd = np.random.random_sample()
    if rnd < prob:
        random_angle = np.random.randint(angle[0], angle[1])
        return image.rotate(random_angle)
    return image

def randomColor(image, prob=0.7, factor=(1, 90)):
    rnd = np.random.random_sample()
    if rnd < prob:
        # Factor 1.0 always returns a copy of the original image,
        # lower factors mean less color (brightness, contrast, etc), and higher values more
        random_factor = np.random.randint(2, 18) / 10.
        color_image = ImageEnhance.Color(image).enhance(random_factor)
        random_factor = np.random.randint(5, 18) / 10.
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
        random_factor = np.random.randint(5, 18) / 10.
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
        random_factor = np.random.randint(2, 18) / 10.
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)
    return image

def randomGaussian(image, prob=0.5, mean=0, sigma=10):
    rnd = np.random.random_sample()
    if rnd < prob:
        img_array = np.asarray(image)
        noisy_img = img_array + np.random.normal(mean, sigma, img_array.shape)
        noisy_img = np.clip(noisy_img, 0, 255)

        return Image.fromarray(np.uint8(noisy_img))
    return image
