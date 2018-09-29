import pandas as pd
import numpy as np

import keras
from keras.preprocessing.text import Tokenizer
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras.callbacks import Callback
from keras.layers import Embedding, Input, Dense, LSTM, GRU, Bidirectional, TimeDistributed
from keras import backend as K
from keras.models import Model

from sklearn.metrics import roc_auc_score

import input as dataset
import tensorflow as tf
from network.layers import AttentionDecoder
import input as data
from utils import helpers
from network.optimizer import Optimizer
from network.evaluation import Metrics
from keras.utils import plot_model
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

from string import punctuation
from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

class VisualATT:
    def __init__(self,
                vocab_size,
                max_length,
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
                early,
                return_probabilities):
        self.vocab_size = vocab_size
        self.max_length = max_length
        #self.model = Sequential()
        # self.model.add(Embedding(vocab_size+1, 100, input_length=max_length,
        #                     trainable = True,
        #                     mask_zero=True))
        # self.model.add(Bidirectional(LSTM(100, return_sequences=True),
        #                             merge_mode='concat',
        #                             trainable=True))
        # self.model.add(AttentionDecoder(100,
        #                         name='attention_decoder_1',
        #                         output_dim=2,
        #                         return_probabilities=return_probabilities,
        #                         trainable=True))
        # self.model.add(Dense(2, activation='softmax'))


        input_ = Input(shape=(max_length,), dtype='float32')
        input_embed = Embedding(vocab_size+1, 100,
                                input_length=max_length,
                                trainable=True,
                                mask_zero=True,
                                name='OneHot')(input_)

        rnn_encoded = Bidirectional(LSTM(100, return_sequences=True),
                                    name='bidirectional_1',
                                    merge_mode='concat',
                                    trainable=True)(input_embed)

        y_hat = AttentionDecoder(units =100,
                                name='attention_decoder_1',
                                output_dim=2,
                                return_sequence=True,
                                return_probabilities=return_probabilities,
                                trainable=True)(rnn_encoded)
        dense = Dense(2, activation='softmax')(y_hat)
        self.model = Model(inputs = input_, outputs = dense)

        plot_model(self.model, to_file='modelHATT.png')
        
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
    
    def Visual(self):
        input_ = Input(shape=(self.max_length,), dtype='float32')
        input_embed = Embedding(self.vocab_size+1, 100,
                                input_length=self.max_length,
                                trainable=True,
                                mask_zero=True,
                                name='OneHot')(input_)

        rnn_encoded = Bidirectional(LSTM(100, return_sequences=True),
                                    name='bidirectional_1',
                                    merge_mode='concat',
                                    trainable=True)(input_embed)

        y_hat = AttentionDecoder(units=100,
                                name='attention_decoder_1',
                                output_dim=2,
                                return_probabilities=True,
                                return_attention=True,
                                trainable=True)(rnn_encoded)
        return Model(inputs = input_, outputs = y_hat)

    def run(self):
        
        self.model.compile(loss=self.loss_function,
                      optimizer='rmsprop',
                      metrics=['acc'])
        return self.model.fit(self.X_train, 
                    self.Y_train, 
                    validation_data=(self.X_test, self.Y_test), 
                    epochs=self.epochs, 
                    batch_size=self.batch_size,
                    callbacks = [self.tensorBoard,self.metrics,self.early])

