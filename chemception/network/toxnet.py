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
import keras.layers
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
                    opt,
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
                    smilesnetmodel,
                    toxceptionmodel,
                    classes=2):
        (imageInput, chemception) = Chemception(n,
                                    inputSize,
                                    X_trainC,
                                    Y_trainC,
                                    X_testC,
                                    Y_testC,
                                    None,
                                    None, 
                                    None,
                                    None,
                                    None,
                                    None,
                                    None,
                                    None,
                                    None,
                                    None,
                                    None,
                                    True,
                                    classes=classes).Concat()
        (textInput,toxtext) = VisualATT( vocab_size,
                                            max_size,
                                            X_trainV,
                                            Y_trainV,
                                            X_testV,
                                            Y_testV,
                                            None,
                                            None,
                                            None,
                                            None,
                                            None,
                                            None,
                                            None,
                                            metrics,
                                            None,
                                            None,
                                            False,classes=classes).Concat()
        mergedOut = keras.layers.concatenate([chemception,toxtext])
        partial = keras.models.Model(inputs = [imageInput,textInput], outputs = mergedOut)
        print('Loading models')
        partial.load_weights(toxceptionmodel,by_name=True)
        partial.load_weights(smilesnetmodel,by_name=True)
        print('Loading end')
        print("Predicting the input")
        outputInput_train = partial.predict({'image_input': X_trainC, 'text_input': X_trainV},verbose=1)
        outputInput_test = partial.predict({'image_input': X_testC, 'text_input': X_testV},verbose=1)
        print('Prediction end')
        
        input_out = Input(shape = (312,),name='image_input')
        firstHidden = keras.layers.Dense(200,name='First_dense')(input_out)
        act = keras.layers.Activation('relu',name='First_act')(firstHidden)
        drop = keras.layers.Dropout(0.15,name='First_drop')(act)
        
        for i in range(1,3):
            firstHidden = keras.layers.Dense(200,name='First_dense')(drop)
            act = keras.layers.Activation('relu',name='First_act')(firstHidden)
            drop = keras.layers.Dropout(0.15,name='First_drop')(act)

        secondHidden = Dense(100,name='Second_dense')(drop)
        act = Activation('relu',name='Second_act')(secondHidden)
        drop = Dropout(0.15,name='Second_drop')(act)

        thirdHidden = Dense(2,name='Thrid_dense')(drop)
        act = Activation('sigmoid',name='output')(thirdHidden)

        self.model = keras.models.Model(inputs = input_out, outputs = act)
        print(self.model.summary())
        keras.utils.plot_model(self.model, to_file='modelToxNet.png')
        self.X_train = outputInput_train
        self.X_test = outputInput_test

        self.Y_train = Y_trainC
        self.Y_test = Y_testC

        self.learning_rate = learning_rate
        self.opt = opt

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
                      optimizer=self.opt,
                      metrics=['acc'])
        return self.model.fit(self.X_train,
                    self.Y_train,
                    validation_data=(self.X_test, self.Y_test), 
                    epochs=self.epochs, 
                    batch_size=self.batch_size,
                    callbacks = [self.tensorBoard,self.metrics,self.early])