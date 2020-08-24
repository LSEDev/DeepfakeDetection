"""
VideoFrameGenerator - Simple Generator
--------------------------------------
A simple frame generator that takes distributed frames from
videos. Inspired by https://github.com/metal3d/keras-video-generators
"""

import os
import glob
import numpy as np
import cv2 as cv
from math import floor
import logging
import re

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

log = logging.getLogger()

from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import \
    ImageDataGenerator, img_to_array


class VideoFrameGenerator(Sequence):
    """
    Create a generator that return batches of frames from video
    - rescale: float fraction to rescale pixel data (commonly 1/255.)
    - nb_frames: int, number of frames to return for each sequence
    - classes: list of str, classes to infer
    - batch_size: int, batch size for each loop
    - shape: tuple, target size of the frames
    - shuffle: bool, randomize files
    - transformation: ImageDataGenerator with transformations
    - split: float, factor to split files and validation
    - nb_channel: int, 1 or 3, to get grayscaled or RGB images
    - glob_pattern: string, directory path with '{classname}' inside that \
        will be replaced by one of the class list
        
    You may use the "classes" property to retrieve the class list afterward.
    The generator has that properties initialized:
    - classes_count: number of classes that the generator manages
    - files_count: number of video that the generator can provides
    - classes: the given class list
    - files: the full file list that the generator will use, this \
        is usefull if you want to remove some files that should not be \
        used by the generator.
    """

    def __init__(
            self,
            rescale=1/255.,
            nb_frames: int = 5,
            classes: list = None,
            batch_size: int = 16,
            target_shape: tuple = (224, 224),
            shuffle: bool = True,
            transformation: ImageDataGenerator = None,
            nb_channel: int = 3,
            glob_pattern: str = './videos/{classname}/*.avi',
            *args,
            **kwargs):

        # deprecation
        if 'split' in kwargs:
            log.warn("Warning, `split` argument is replaced by `split_val`, "
                     "please condider to change your source code."
                     "The `split` argument will be removed "
                     "in future releases.")
            split_val = float(kwargs.get('split'))

        self.glob_pattern = glob_pattern

        # should be only RGB or Grayscale
        assert nb_channel in (1, 3)

        if classes is None:
            classes = self._discover_classes()

        # we should have classes
        if len(classes) == 0:
            log.warn("You didn't provide classes list or "
                     "we were not able to discover them from "
                     "your pattern.\n"
                     "Please check if the path is OK, and if the glob "
                     "pattern is correct.\n"
                     "See https://docs.python.org/3/library/glob.html")

        # shape size should be 2
        assert len(target_shape) == 2
        classes.sort()

        self.rescale = rescale
        self.classes = classes
        self.batch_size = batch_size
        self.nbframe = nb_frames
        self.shuffle = shuffle
        self.target_shape = target_shape
        self.nb_channel = nb_channel
        self.transformation = transformation

        self._random_trans = []
        self.files = []

        for cls in classes:
            self.files += glob.glob(glob_pattern.format(classname=cls))

        # build indexes
        self.files_count = len(self.files)
        self.indexes = np.arange(self.files_count)
        self.classes_count = len(classes)

        # to initialize transformations and shuffle indices
        if 'no_epoch_at_init' not in kwargs:
            self.on_epoch_end()

        kind = "train"

        self._current = 0
        print("Total data: %d classes for %d files for %s" % (
            self.classes_count,
            self.files_count,
            kind))
    
    # not so relevant
    def _discover_classes(self):
        pattern = os.path.realpath(self.glob_pattern)
        pattern = re.escape(pattern)
        pattern = pattern.replace('\\{classname\\}', '(.*?)')
        pattern = pattern.replace('\\*', '.*')

        files = glob.glob(self.glob_pattern.replace('{classname}', '*'))
        classes = set()
        for f in files:
            f = os.path.realpath(f)
            cl = re.findall(pattern, f)[0]
            classes.add(cl)

        return list(classes)

    def next(self):
        """ Return next element"""
        elem = self[self._current]
        self._current += 1
        if self._current == len(self):
            self._current = 0
            self.on_epoch_end()

        return elem

    def on_epoch_end(self):
        """ Called by Keras after each epoch """

        if self.transformation is not None:
            self._random_trans = []
            for _ in range(self.files_count):
                self._random_trans.append(
                    self.transformation.get_random_transform(self.target_shape)
                )

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return int(np.floor(self.files_count / self.batch_size))

    def __getitem__(self, index):
        classes = self.classes
        shape = self.target_shape
        nbframe = self.nbframe

        labels = []
        images = []

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        transformation = None

        for i in indexes:
            # prepare a transformation if provided
            if self.transformation is not None:
                transformation = self._random_trans[i]

            video = self.files[i]
            classname = self._get_classname(video)

            # create a label array and set 1 to the right column
            label = np.zeros(len(classes))
            col = classes.index(classname)
            label[col] = 1.
            
            frames = self._get_frames(
                video,
                nbframe,
                shape)
            # apply transformation
            if transformation is not None:
                frames = [self.transformation.apply_transform(
                    frame, transformation) for frame in frames]

            # add the sequence in batch
            images.append(frames)
            labels.append(label)

        return np.array(images), np.array(labels)

    def _get_classname(self, video: str) -> str:
        """ Find classname from video filename following the pattern """

        # work with real path
        video = os.path.realpath(video)
        pattern = os.path.realpath(self.glob_pattern)

        # remove special regexp chars
        pattern = re.escape(pattern)

        # get back "*" to make it ".*" in regexp
        pattern = pattern.replace('\\*', '.*')

        # use {classname} as a capture
        pattern = pattern.replace('\\{classname\\}', '(.*?)')

        # and find all occurence
        classname = re.findall(pattern, video)[0]
        return classname
    
    def load_img(self, path, grayscale=False, target_size=None):
        """Loads an image into PIL format.
        # Arguments
            path: Path to image file
            grayscale: Boolean, whether to load the image as grayscale.
            target_size: Either `None` (default to original size)
                or tuple of ints `(img_height, img_width)`.
        # Returns
            A PIL Image instance.
        # Raises
            ImportError: if PIL is not available.
        """
        if pil_image is None:
            raise ImportError('Could not import PIL.Image. '
                              'The use of `array_to_img` requires PIL.')
        img = pil_image.open(path)
        if grayscale:
            if img.mode != 'L':
                img = img.convert('L')
        else:
            if img.mode != 'RGB':
                img = img.convert('RGB')
        if target_size:
            hw_tuple = (target_size[1], target_size[0])
            if img.size != hw_tuple:
                img = img.resize(hw_tuple)
        return img
    
    
    def _get_frames(self, video, nbframe, shape):
        frames = []
        for fname in sorted(os.listdir(video)):
            if fname != '.ipynb_checkpoints':
                path = video + '/' + fname
                img = self.load_img(path, False, shape)
                frame = img_to_array(img) * self.rescale
                frames.append(frame)
            
                if len(frames) == nbframe:
                    break
        
        # Could write recursive function, but this should do the trick
        if len(frames) != nbframe:
            for fname in sorted(os.listdir(video)):
                if fname != '.ipynb_checkpoints':
                    path = video + '/' + fname
                    img = self.load_img(path, False, shape)
                    frame = img_to_array(img) * self.rescale
                    frames.append(frame)
        
                    if len(frames) == nbframe:
                        break

        return np.array(frames)