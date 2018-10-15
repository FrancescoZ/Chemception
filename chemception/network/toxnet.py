from network import Chemception
from network import VisualATT
import input as data
from keras.layers import *
from keras.preprocessing.sequence import pad_sequences
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from keras.preprocessing.sequence import pad_sequences
import input as data



from utils import helpers
from utils import Visualizer
from utils import visualize
from network.optimizer import Optimizer
from network.evaluation import Metrics

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
import keras.backend as K

import os
import sys
import time
import statistics
import shutil

import numpy as nu

from sklearn.model_selection import train_test_split
import input as data
import numpy as np
class ToxNet:

    def __init__(self,
                    n,
                    inputSize, 
                   X_trainC,
                    Y_trainC,
                    X_testC,
                    Y_testC,
                    X_trainV,
                    Y_trainV,
                    X_testV,
                    Y_testV,
                    learning_rate,
                    rho,
                    epsilon,
                    epochs,
                    loss_function,
                    log_dir,
                    batch_size,
                    data_augmentation,
                    metrics,
                    tensorBoard,
                    early,
                    vocab_size,
                    max_size,
                    classes=2):
        (imageInput, chemception) = Chemception(n,
                                    inputSize,
                                    X_trainC,
                                    Y_trainC,
                                    X_testC,
                                    Y_testC,
                                     metrics,
                                    tensorBoard,
                                    early, 
                                    learning_rate,
                                    rho,
                                    epsilon,
                                    epochs,
                                    loss_function,
                                    log_dir,
                                    batch_size,
                                    data_augmentation,
                                    True,
                                    classes=classes).Concat()
        (textInput,toxtext) = VisualATT( vocab_size,
                                            max_size,
                                            X_trainV,
                                            Y_trainV,
                                            X_testV,
                                            Y_testV,
                                            learning_rate,
                                            rho,
                                            epsilon,
                                            epochs,
                                            loss_function,
                                            log_dir,
                                            batch_size,
                                            metrics,
                                            tensorBoard,
                                            early,
                                            False,classes=classes).Concat()
        print(chemception.shape)
        dense = Dense(classes, activation='softmax',name ='dense_smile')(toxtext)
        mergedOut = keras.layers.concatenate([chemception,toxtext])

        firstHidden = keras.layers.Dense(500,name='First_dense')(mergedOut)
        act = keras.layers.Activation('relu',name='First_act')(firstHidden)
        drop = keras.layers.Dropout(0.4,name='First_drop')(act)

        secondHidden = Dense(300,name='Second_dense')(drop)
        act = Activation('relu',name='Second_act')(secondHidden)
        drop = Dropout(0.4,name='Second_drop')(act)
        
        thirdHidden = Dense(2,name='Thrid_dense')(drop)
        act = Activation('softmax',name='output')(thirdHidden)

        self.model = keras.models.Model(inputs = [imageInput,textInput], outputs = act)
        print(self.model.summary())
        keras.utils.plot_model(self.model, to_file='modelToxNet.png')
        self.X_trainC = X_trainC
        self.Y_trainC = Y_trainC
        self.X_testC = X_testC
        self.Y_testC = Y_testC

        self.X_trainV = X_trainV
        self.Y_trainV = Y_trainV
        self.X_testV = X_testV
        self.Y_testV = Y_testV

        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.epochs = epochs

        self.loss_function = loss_function

        self.log_dir = log_dir
        self.batch_size = batch_size

        self.data_augmentation = data_augmentation

        self.metrics = metrics
        self.tensorBoard = tensorBoard
        self.early = early
        print(self.model.summary())
    
    def get_output_layer(self, model, layer_name):
        # get the symbolic outputs of each "key" layer (we gave them unique names).
        layer_dict = dict([(layer.name, layer) for layer in model.layers])
        layer = layer_dict[layer_name]
        return layer

    def run(self):
        self.model.compile(loss=self.loss_function,
                      optimizer='rmsprop',
                      metrics=['acc'])
        return self.model.fit(
                    {'image_input': self.X_trainC, 'text_input': self.X_trainV},
                    {'dense_smile': self.Y_trainV, 'output': self.Y_trainC},
                    validation_data=({'image_input': self.X_testC, 'text_input': self.X_testV}, {'dense_smile': self.Y_testV, 'output': self.Y_testC}), 
                    epochs=self.epochs, 
                    batch_size=self.batch_size,
                    callbacks = [self.tensorBoard,self.metrics,self.early])