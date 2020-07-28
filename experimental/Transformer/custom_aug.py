#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!pip install imgaug

from imgaug import augmenters as iaa
import random
import numpy as np
    
def normalize(x):
    return (x / 255.0).copy()
    
def denormalize(x):
    return (x * 255.0).copy()

# BLOCKS
#################################################################################################################################
def perform_crop(x):
    magnitude = 0.634
    x = iaa.Crop(px=(0, int(magnitude * 32))).augment_image(x)
    return x

def perform_additive_gaussian_noise(x):
    magnitude = 0.25
    x = iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, magnitude * 255), per_channel=0.5).augment_image(x)
    return x

def perform_gamma_contrast(x):
    magnitude = 0.75
    X_norm = normalize(x)
    X_aug_norm = iaa.GammaContrast(magnitude * 1.75).augment_image(X_norm)
    x = denormalize(X_aug_norm)
    return x

def perform_elastic_transform(x):
    magnitude = 0.1
    X_norm = normalize(x)
    X_norm2 = (X_norm * 2) - 1
    X_aug_norm2 = iaa.ElasticTransformation(alpha=(0.0, max(0.5, magnitude * 300)), sigma=5.0).augment_image(X_norm2)
    X_aug_norm = (X_aug_norm2 + 1) / 2
    x = denormalize(X_aug_norm)
    return x

def perform_add_to_hue_and_saturation(x):
    magnitude = 0.5
    x = iaa.AddToHueAndSaturation((int(-45 * magnitude), int(45 * magnitude))).augment_image(x.astype(np.uint8))
    x = x.astype(float)
    return x

def perform_coarse_salt_and_pepper(x):
    magnitude = 0.25
    x = iaa.CoarseSaltAndPepper(p=0.2, size_percent=magnitude).augment_image(x)
    return x


def perform_gaussian_blur(x):
    magnitude = 3.0/25.0
    x = iaa.GaussianBlur(sigma=(0, magnitude * 25.0)).augment_image(x)
    return x

def perform_sharpen(x):
    magnitude = 0.5
    x = iaa.Sharpen(alpha=(0, 1.0), lightness=(0.50, 5 * magnitude)).augment_image(x)
    return x

def perform_brighten(x):
    magnitude = 1.0
    x = iaa.Add((int(-40 * magnitude), int(40 * magnitude)), per_channel=0.5).augment_image(x)
    return x

def perform_all_channel_histogram_equalization(x):
    x = iaa.AllChannelsHistogramEqualization().augment_image(x.astype(np.uint8))
    x = x.astype(float)
    return x


def perform_perspective_transform(x):
    magnitude = 0.15
    X_norm = normalize(x)
    X_aug_norm = iaa.PerspectiveTransform(scale=(0.01, max(0.02, magnitude))).augment_image(X_norm) 
    # first scale param must be larger
    np.clip(X_aug_norm, 0.0, 1.0, out=X_aug_norm)
    x = denormalize(X_aug_norm)
    return x

def perform_grayscale(x):
    magnitude = 1.0
    x = iaa.Grayscale(alpha=(0.25, magnitude)).augment_image(x.astype(np.uint8))
    x = x.astype(float)
    return x
#################################################################################################################################

# JOINT FUNCTION
#################################################################################################################################
def joint_function(x):
    
    # No augmentations with probability (1/5)
    #############################################################
    chance = np.random.random()
    if chance <= 1/5:
        return x
    #############################################################
    
    # DEEPAUGMENT'S OPTIMAL POLICY (2/5)
    #############################################################
    elif chance <= 3/5:
    # either a colour-based double augmentation or one out of four,
    # which all consist of one geometry based and one noise-based
    # augmentations
    # all with equal probability of 8%
    
        # Colour-based
        if np.random.random() <= 1/5:
            x = perform_gamma_contrast(x)
            x = perform_add_to_hue_and_saturation(x)
        
        # combine one noise-based and one geometry-based method
        else:
        
            # noise
            if np.random.random() <= 1/2:
                x = perform_coarse_salt_and_pepper(x)
            else:
                x = perform_additive_gaussian_noise(x)

            # geometry-based methods
            if np.random.random() <= 1/2:
                x = perform_crop(x)
            else:
                x = perform_elastic_transform(x)
    #############################################################
    
    # ADDITIONAL AUGMENTATIONS (2/5)
    #############################################################
    else:
    # secondary transformations all with equal probability 6.67%
        second_chance = np.random.random()
        
        if  second_chance <= 1/6:
            x = perform_gaussian_blur(x)
        elif second_chance <= 1/3:
            x = perform_sharpen(x)
        elif second_chance <= 1/2:
            x = perform_brighten(x)
        elif second_chance <= 2/3:
            x = perform_all_channel_histogram_equalization(x)
        elif second_chance <= 5/6:
            x = perform_perspective_transform(x)
        else:
            x = perform_grayscale(x)
    #############################################################

    return x
#################################################################################################################################