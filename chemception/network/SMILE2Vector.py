
import numpy as nu
import input as data
from utils import helpers
from utils import constant
from network.evaluation import Metrics

import keras
import keras.backend as K

import os
import sys
import time
import statistics
import shutil

from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from keras.utils import to_categorical
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.optimizers import SGD
import keras
from keras.utils import plot_model


import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
import keras.backend as K

class SMILE2Vector:

    def __init__(self,
                    n,
                    X_train,
                    Y_train,
                    X_test,
                    Y_test,
                    learning_rate,
                    rho,
                    epsilon,
                    epochs,
                    loss_function,
                    log_dir,
                    batch_size,
                    metrics,
                    tensorBoard,
                    early):
        
        embedding_vecor_length = 60
        top_words                = 157
        max_review_length        = 500
        self.model = Sequential()
        self.model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length,mask_zero=True))
        self.model.add(LSTM(384, return_sequences=True))
        self.model.add(LSTM(384))
        self.model.add(Dense(2, activation='softmax'))
    
        # truncate and pad input sequences
        plot_model(self.model, to_file='modelSMILE2Vect.png')

        X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
        X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.epochs = epochs

        self.loss_function = loss_function

        self.log_dir = log_dir
        self.batch_size = batch_size

        self.metrics = metrics
        self.tensorBoard = tensorBoard
        self.early = early
        print(self.model.summary())
    
    def run(self):
        self.model.compile(loss=self.loss_function, optimizer='adam', metrics=['accuracy'])
        return self.model.fit(self.X_train, 
                    self.Y_train, 
                    validation_data=(self.X_test, self.Y_test), 
                    epochs=self.epochs, 
                    batch_size=self.batch_size,
                    callbacks = [self.tensorBoard,self.metrics,self.early])
    def printModel(self):
        plot_model(self.model, to_file='modelSMILE2Vect.png')
