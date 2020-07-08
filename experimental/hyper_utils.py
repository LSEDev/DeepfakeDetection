import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
import tensorflow.keras.backend as K
import os
import glob
from tensorflow.keras.callbacks import *
import warnings
import logging
from IPython.display import clear_output
import sys

class PrintLR(tf.keras.callbacks.Callback):
    '''Prints the LR to be used for the batch before the batch is fed into the network for training.'''
    
    def on_train_batch_begin(self, batch, logs=None):
        LR = K.get_value(self.model.optimizer.lr)
        if type(LR)==tf.python.keras.optimizer_v2.learning_rate_schedule.CosineDecay:
            print("\nLearning rate:", LR.__call__(self.model.optimizer.iterations))
        else:
            print("\nLearning rate:", LR) 
        print(K.get_value(self.model.optimizer.lr))
        
class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

  Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
  """

    def __init__(self, schedule):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(K.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        # Set the value back to the optimizer before this epoch starts
        K.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))


# LR_SCHEDULE = [
#     # (epoch to start, learning rate) tuples
#     (1, tf.keras.experimental.CosineDecay(initial_learning_rate=lr_rate, decay_steps=5000000, alpha=0.1)),
#     (6, 0.01),
#     (9, 0.005),
#     (12, 0.001)]


# def lr_schedule(epoch, lr):
#     """Helper function to retrieve the scheduled learning rate based on epoch."""
#     if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
#         return lr
#     for i in range(len(LR_SCHEDULE)):
#         if epoch == LR_SCHEDULE[i][0]:
#             return LR_SCHEDULE[i][1]
#     return lr

def lr_schedule(epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    if epoch < 5:
        return lr/(6-epoch)
    elif epoch == 5:
        return tf.keras.experimental.CosineDecay(initial_learning_rate=lr, decay_steps=5000000, alpha=0.1)
    return lr


# model = get_model()
# model.fit(
#     x_train,
#     y_train,
#     batch_size=64,
#     steps_per_epoch=5,
#     epochs=15,
#     verbose=0,
#     callbacks=[
#         LossAndErrorPrintingCallback(),
#         CustomLearningRateScheduler(lr_schedule),
#     ],
# )
    
    