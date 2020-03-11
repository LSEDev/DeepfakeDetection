import cv2
import sys
import os
from os.path import join
import argparse
import subprocess
from tqdm import tqdm

DATASET_PATHS = {
    'original': 'original_sequences/youtube',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceSwap': 'manipulated_sequences/FaceSwap'
    'NeuralTextures' : 'manipulated_sequences/NeuralTextures'
}
COMPRESSION = ['c0', 'c23', 'c40']

# specify yourself
CASCADE_PATH = "/Users/superlaut/face_scrapper/face_scrapper/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml"

FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)


def extract_face(data_path, output_path, imgname):
    #os.makedirs(output_path, exist_ok=True)
    img = cv2.imread(data_path)
    if (img is None):
        print("Can't open image file")
        return 0
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(img, 1.1, 3, minSize=(100, 100))
    if (faces is None):
        print('Failed to detect face')
        return 0

    facecnt = len(faces)
    print("Detected faces: %d" % facecnt)
    
    i = 0
    height, width = img.shape[:2]

    for (x, y, w, h) in faces:
        r = max(w, h) / 2
        centerx = x + w / 2
        centery = y + h / 2
        nx = int(centerx - r)
        ny = int(centery - r)
        nr = int(r * 2)

        faceimg = img[int(0.8*ny):int(1.2*(ny+nr)), 
                      int(0.8*nx):int(1.2*(nx+nr))]
        lastimg = cv2.resize(faceimg, (32, 32))
        # counter not necessary when absolutely sure only
        # one face included
        i += 1
        #cv2.imwrite("image%d.jpg" % i, lastimg)
        if i == 1:
            cv2.imwrite(join(output_path, imgname),lastimg)
        else: 
            return 0
        
def extract_method_images(data_path, dataset, compression):
    images_path = join(data_path, DATASET_PATHS[dataset], compression, 'images')
    cropped_path = join(data_path, DATASET_PATHS[dataset], compression, 'croppedfaces')
    
    for image_folder in os.listdir(images_path):
        if image_folder != '.DS_Store':
            cropped_folder = str(image_folder) + '_faces'
            os.makedirs(join(cropped_path, cropped_folder), exist_ok=True)
            count = 0
            for image in tqdm(os.listdir(join(images_path, image_folder))):
                caption = str(count) + '.jpg'
                count += 1
                extract_face(join(images_path, image_folder, image),
                             join(cropped_path, cropped_folder), 
                             caption)

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--data_path', '-dp', type=str)
    p.add_argument('--dataset', '-d', type=str,
                   choices=list(DATASET_PATHS.keys()) + ['all'],
                   default='all')
    p.add_argument('--compression', '-c', type=str, choices=COMPRESSION,
                   default='c0')
    args = p.parse_args()

    if args.dataset == 'all':
        for dataset in DATASET_PATHS.keys():
            args.dataset = dataset
            extract_method_images(**vars(args))
    else:
        extract_method_images(**vars(args))