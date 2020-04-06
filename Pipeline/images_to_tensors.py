#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 18:55:44 2020

@author: gregmeyer
"""

import cv2
import glob
import numpy as np
import pickle 

path = "/Users/gregmeyer/Desktop/LSE/Capstone/Test"

def get_images(img_path, ori_img_path, manip_img_path):
    '''Outputs a tensor of image pixels in a specificied path as well as a count
    of the non-manipualted images
    Resizes images to all be the same size'''

    image_data_list = []
    ori_count=0
    for i in [ori_img_path, manip_img_path]:
        for j in glob.glob(img_path + '/' + i + '/*'):
            for each_img in glob.glob(j + '/*'):
                input_img = cv2.imread(each_img)
                input_img_resize=cv2.resize(input_img,(71,71)) # for future modification
                image_data_list.append(input_img_resize)
                if i==ori_img_path:
                    ori_count+=1

    return np.array(image_data_list), ori_count

x_data, ori_count = get_images(path, 'original_sequences/croppedfaces',
                               'manipulated_sequences/croppedfacesDeepfake')

# Save image tensors to disk
with open(path + '/Pickles/image_tensors.pickle', 'wb') as handle:
    pickle.dump(x_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save ori count to disk
with open(path + '/Pickles/ori_count.pickle', 'wb') as handle:
    pickle.dump(ori_count, handle, protocol=pickle.HIGHEST_PROTOCOL)