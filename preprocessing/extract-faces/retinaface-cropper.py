from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import numpy as np
import tensorflow as tf
import time
import glob

from modules.models import RetinaFaceModel
from modules.utils import (set_memory_growth, load_yaml, draw_bbox_landm,
                           pad_input_image, recover_pad_output)


flags.DEFINE_string('cfg_path', './configs/retinaface_res50.yaml',
                    'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_string('path', '', 'path to directory')
flags.DEFINE_float('iou_th', 0.4, 'iou threshold for nms')
flags.DEFINE_float('score_th', 0.5, 'score threshold for nms')
flags.DEFINE_float('scaling_factor', 1.25, 'scaling factor for cropping faces')
flags.DEFINE_float('min_scaling_factor', 1.0, 'smallest allowed scaling factor for cropping faces')
flags.DEFINE_float('threshold_prob', 0.5, 'threshold probability of an element being a face')
flags.DEFINE_integer('max_iter', 10, 'max number of times the scaled up bounding box is shifted')

def main(_argv):
    
    # FUNCTIONS FOR CROPPING
    #####################################################################################################
    def bounding_box(img, ann, img_height, img_width):
        x1, y1, x2, y2 = int(ann[0] * img_width), int(ann[1] * img_height), \
                         int(ann[2] * img_width), int(ann[3] * img_height)
        return x1, y1, x2, y2
    
    def calc_points(x, y, side):
        return int(x - side/2), int(x + side/2), int(y - side/2), int(y + side/2)
                
    def adjust_points(x_center, y_center, original_longest, scaling_factor,
                      min_scaling_factor):          
        factors = np.arange(scaling_factor, min_scaling_factor - 0.04, -0.05)         
        for factor in factors:
            # calculate nex points
            x1, x2, y1, y2 = calc_points(x_center, y_center, int(original_longest * factor))
                        
            for i in range(FLAGS.max_iter):
                if x1 < 0: x2 -= x1; x1 = 0 
                if y1 < 0: y2 -= y1; y1 = 0
                if x2 > img_raw.shape[1]: x1 -= x2; x2 = img_raw.shape[1]
                if y2 > img_raw.shape[0]: y1 -= y2; y2 = img_raw.shape[0]

                if x1 >= 0 and y1 >= 0 and x2 <= img_raw.shape[1] and y2 <= img_raw.shape[0]:
                    return x1, x2, y1, y2, True
                
        print("Not cropping", img_path, "due to a problem with a cropping square box")
        return x1, x2, y1, y2, False
    
    def get_dim(lst):
        return [(lst[3]-lst[1]) * (lst[2]-lst[0])]
    
    def get_max(outputs, lst):
        area = [i[0] for i in lst]
        prob = [i[1] for i in lst]
        max_area_index = set([i for i, j in enumerate(area) if j == max(area)])
        max_prob_index = set([i for i, j in enumerate(prob) if j == max(prob)])
        indecies = list(max_area_index.intersection(max_prob_index))
        if len(indecies) >= 1: return [outputs[indecies[0]]]
        elif len(indecies) == 0: # if there is a mismatch, return the largest element
            if len(list(max_area_index)) >= 1: return [outputs[list(max_area_index)[0]]]
            else: # precautionary because there should always be at least one face
                print("Not cropping", img_path, "due to a problem with returning the largest element")
                return []
    #####################################################################################################
    
    # MODEL
    #####################################################################################################
    # initialisation
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()
    cfg = load_yaml(FLAGS.cfg_path)

    # define network
    model = RetinaFaceModel(cfg, training=False, iou_th=FLAGS.iou_th,
                            score_th=FLAGS.score_th)

    # load checkpoint
    checkpoint_dir = './checkpoints/' + cfg['sub_name']
    checkpoint = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("[*] load ckpt from {}.".format(
            tf.train.latest_checkpoint(checkpoint_dir)))
    else:
        print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
        exit()
    #####################################################################################################
    
    # CROPPING
    #####################################################################################################
    # check if the path exits
    if not os.path.exists(FLAGS.path):
        print(f"cannot find the specified path from {FLAGS.path}")
        exit()
    
    # make a corresponding directory
    try:
        os.mkdir(FLAGS.path.replace("images", "cropped_images"))
    except FileExistsError:
        print(FLAGS.path.replace("images", "cropped_images"), "already exists")
        
    # eget subdirectories within the specified folder
    subdirectories = [FLAGS.path+'/'+i for i in os.listdir(FLAGS.path) \
                      if os.path.isdir(FLAGS.path+'/'+i)]
    
    # loop through each folder
    for subdir in sorted(subdirectories):
       
        # create corresponding folders for cropped data and get all images in a given folder
        if 'original' in subdir: x = 3
        else: x = 7
            
        try:
            os.mkdir(subdir.replace("images", "cropped_images"))
            images_lst = glob.glob(subdir + "/*.png")
            cropped_images_lst = []
            print(subdir[len(subdir)-x:len(subdir)])
            
        except FileExistsError:
            # count number of existing images in this subdirectory, if same as original, skip
            images_lst = glob.glob(subdir + "/*.png")
            cropped_images_lst = glob.glob(subdir.replace("images", "cropped_images") + "/*.png")
            cropped_images_lst = [e[len(e)-8:len(e)] for e in cropped_images_lst]
            
            if len(images_lst) == len(cropped_images_lst):
                print(subdir[len(subdir)-x:len(subdir)], "has already been generated")
                continue
            else:
                print(subdir[len(subdir)-x:len(subdir)])

        # loop through each image in a given folder
        for img_path in sorted(images_lst):
            
            if img_path[len(img_path)-8:len(img_path)] in cropped_images_lst:
                continue
            
            img_raw = cv2.imread(img_path)
            img_height_raw, img_width_raw, _ = img_raw.shape
            img = np.float32(img_raw.copy())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # pad input image (unmatched shape problem), run model, recover padding effect
            img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))
            outputs = model(img[np.newaxis, ...]).numpy()
            outputs = recover_pad_output(outputs, pad_params)
            
            # get rid of elements which are faces with less that threshold probability
            outputs = [i for i in outputs if i[15] >= FLAGS.threshold_prob]

            # flag any images which have no recognised faces in them
            if len(outputs) == 0: print("no faces detected for", img_path)
                
            # if more than one face detected, select the largest and most definite    
            elif len(outputs) > 1:
                f = [list(bounding_box(img_raw, i[0:4], img_height_raw, img_width_raw)) + [i[15]] \
                           for i in outputs]
                f = [get_dim(i[0:4]) + [i[4]] for i in f]
                outputs = get_max(outputs, f)
            
            # keeping as a loop in case we decide to use multiple faces per frame in the future
            # get cropping coordinates and save results
            for prior_index in range(len(outputs)):
                # get the bounding box coordinates
                bb_x1, bb_y1, bb_x2, bb_y2 = bounding_box(img_raw, outputs[prior_index],
                                              img_height_raw, img_width_raw)
                # scale up the magnitude of the longest side
                original_longest = int(max(bb_x2-bb_x1, bb_y2-bb_y1))
                longest = int(original_longest * FLAGS.scaling_factor)
                x_center = int((bb_x1 + bb_x2)/2)
                y_center = int((bb_y1 + bb_y2)/2)
                
                x1, x2, y1, y2, save_image = adjust_points(x_center, y_center, original_longest,
                                                           FLAGS.scaling_factor,
                                                           FLAGS.min_scaling_factor)
              
                if save_image:
                    try:
                        save_img_path = os.path.join(subdir.replace("images", "cropped_images") \
                            + "/" + img_path.replace(subdir + '/', ''))
                        cv2.imwrite(save_img_path, img_raw[y1 : y2, x1 : x2])

                    except:
                        print(img_path, "is not cropped for unknown reasons")
    #####################################################################################################

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
