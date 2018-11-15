# import pandas as pd
# import numpy as np

# import keras
# from keras.preprocessing.text import Tokenizer
# from keras.engine.topology import Layer
# from keras import initializers as initializers, regularizers, constraints
# from keras.callbacks import Callback
# from keras.layers import Embedding, Input, Dense, LSTM, GRU, Bidirectional, TimeDistributed, Dropout
# from keras import backend as K
# from keras.models import Model

# from sklearn.metrics import roc_auc_score

# import input as dataset
# import tensorflow as tf
# from network.layers import AttentionDecoder
# import input as data
# from utils import helpers
# from network.optimizer import Optimizer
# from network.evaluation import Metrics
# from keras.utils import plot_model
# import keras
# from keras.preprocessing.image import ImageDataGenerator
# from keras.optimizers import SGD
# from keras.callbacks import TensorBoard
# import keras.backend as K

# import os
# import sys
# import time
# import statistics
# import shutil

# import numpy as nu

# from sklearn.model_selection import train_test_split

# from string import punctuation
# from os import listdir
# from numpy import array
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Flatten
# from keras.layers import Embedding
# from keras.layers.convolutional import Conv1D
# from keras.layers.convolutional import MaxPooling1D

# from keras.models import Sequential, Model
# from keras.layers import Dropout, Dense, Input
# from keras.optimizers import Adam, Nadam
# from keras.activations import relu, elu, sigmoid
# from keras.losses import binary_crossentropy

# class VisualATT:

#     def __init__(self,
#                 vocab_size,
#                 max_length,
#                 X_train,
#                 Y_train,
#                 X_test,
#                 Y_test,
#                 metrics,
#                 tensorBoard,
#                 early,
#                 learning_rate='',
#                 rho='',
#                 epsilon='',
#                 epochs='',
#                 loss_function='',
#                 log_dir='',
#                 batch_size='',                
#                 return_probabilities='',
#                 classes = 2):
#         self.vocab_size = vocab_size
#         self.max_length = max_length


#         input_ = Input(shape=(max_length,), dtype='float32',name='text_input')
#         input_embed = Embedding(vocab_size+1, 100,
#                                 input_length=max_length,
#                                 trainable=True,
#                                 mask_zero=True,
#                                 name='OneHot_smile')(input_)

#         rnn_encoded = Bidirectional(LSTM(100, return_sequences=True),
#                                     name='bidirectional_smile',
#                                     merge_mode='concat',
#                                     trainable=True)(input_embed)

#         y_hat = AttentionDecoder(units =100,
#                                 name='attention_decoder_smile',
#                                 output_dim=2,
#                                 return_sequence=True,
#                                 return_probabilities=return_probabilities,
#                                 trainable=True)(rnn_encoded)
#         dense = Dense(classes, activation='softmax',name ='dense_smile')(y_hat)
#         self.model = Model(inputs = input_, outputs = dense)

#         plot_model(self.model, to_file='modelHATT.png')
        
#         self.X_train = X_train
#         self.Y_train = Y_train
#         self.X_test = X_test
#         self.Y_test = Y_test

#         self.learning_rate = learning_rate
#         self.rho = rho
#         self.epsilon = epsilon
#         self.epochs = epochs

#         self.loss_function = loss_function

#         self.log_dir = log_dir
#         self.batch_size = batch_size

#         self.metrics = metrics
#         self.tensorBoard = tensorBoard
#         self.early = early
#         self.classes = classes
#         self.opt = 'Adam'
#         print(self.model.summary())
         
#     def Concat(self):
#         input_ = Input(shape=(self.max_length,), dtype='float32',name='text_input')
#         input_embed = Embedding(self.vocab_size+1, 100,
#                                 input_length=self.max_length,
#                                 trainable=True,
#                                 mask_zero=True,
#                                 name='OneHot_smile')(input_)

#         rnn_encoded = Bidirectional(LSTM(100, return_sequences=True),
#                                     name='bidirectional_smile',
#                                     merge_mode='concat',
#                                     trainable=True)(input_embed)

#         y_hat = AttentionDecoder(units =100,
#                                 name='attention_decoder_smile',
#                                 output_dim=2,
#                                 return_sequence=True,
#                                 return_probabilities=True,
#                                 return_attention=True,
#                                 trainable=True)(rnn_encoded)
#         return input_, y_hat
        
#     def Visual(self):
#         input_ = Input(shape=(self.max_length,), dtype='float32',name='text_input')
#         input_embed = Embedding(self.vocab_size+1, 100,
#                                 input_length=self.max_length,
#                                 trainable=True,
#                                 mask_zero=True,
#                                 name='OneHot_smile')(input_)

#         rnn_encoded = Bidirectional(LSTM(100, return_sequences=True),
#                                     name='bidirectional_smile',
#                                     merge_mode='concat',
#                                     trainable=True)(input_embed)

#         y_hat = AttentionDecoder(units =100,
#                                 name='attention_decoder_smile',
#                                 output_dim=2,
#                                 return_sequence=True,
#                                 return_probabilities=True,
#                                 return_attention=True,
#                                 trainable=True)(rnn_encoded)
#         return Model(inputs = input_, outputs = y_hat)

#     def run(self):
#         self.model.compile(loss=self.loss_function,
#                       optimizer=self.opt,
#                       metrics=['acc'])
#         return self.model.fit(self.X_train, 
#                     self.Y_train, 
#                     validation_data=(self.X_test, self.Y_test), 
#                     epochs=self.epochs, 
#                     batch_size=self.batch_size,
#                     callbacks = [self.early,self.metrics,self.tensorBoard])
    
import pandas as pd
import numpy as np

import keras
from keras.preprocessing.text import Tokenizer
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras.callbacks import Callback
from keras.layers import Embedding, Input, Dense, LSTM, GRU, Bidirectional, TimeDistributed, Dropout
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

from keras.models import Sequential, Model
from keras.layers import Dropout, Dense, Input
from keras.optimizers import Adam, Nadam
from keras.activations import relu, elu, sigmoid
from keras.losses import binary_crossentropy

class VisualATT:

    def __init__(self,
                vocab_size,
                max_length,
                X_train,
                Y_train,
                X_test,
                Y_test,
                metrics,
                tensorBoard,
                early,
                learning_rate='',
                rho='',
                epsilon='',
                epochs='',
                loss_function='',
                log_dir='',
                batch_size='',                
                return_probabilities='',
                classes = 2):
        self.vocab_size = vocab_size
        self.max_length = max_length


        input_ = Input(shape=(max_length,), dtype='float32',name='text_input')
        input_embed = Embedding(vocab_size+1, 400,
                                input_length=max_length,
                                trainable=True,
                                mask_zero=True,
                                name='OneHot_smile')(input_)

        rnn_encoded = Bidirectional(LSTM(400, return_sequences=True),
                                    name='bidirectional_smile',
                                    merge_mode='sum',
                                    trainable=True)(input_embed)

        y_hat = AttentionDecoder(units =400,
                                name='attention_decoder_smile',
                                output_dim=2,
                                return_sequence=True,
                                return_probabilities=return_probabilities,
                                trainable=True)(rnn_encoded)
        dense = Dense(classes, activation='softmax',name ='dense_smile')(y_hat)
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
        self.classes = classes
        self.opt = 'Adam'
        print(self.model.summary())
         
    def Concat(self):
        input_ = Input(shape=(self.max_length,), dtype='float32',name='text_input')
        input_embed = Embedding(self.vocab_size+1, 400,
                                input_length=self.max_length,
                                trainable=True,
                                mask_zero=True,
                                name='OneHot_smile')(input_)

        rnn_encoded = Bidirectional(LSTM(400, return_sequences=True),
                                    name='bidirectional_smile',
                                    merge_mode='sum',
                                    trainable=True)(input_embed)

        y_hat = AttentionDecoder(units =400,
                                name='attention_decoder_smile',
                                output_dim=2,
                                return_sequence=True,
                                return_probabilities=True,
                                return_attention=True,
                                trainable=True)(rnn_encoded)
        return input_, y_hat
        
    def Visual(self):
        input_ = Input(shape=(self.max_length,), dtype='float32',name='text_input')
        input_embed = Embedding(self.vocab_size+1, 400,
                                input_length=self.max_length,
                                trainable=True,
                                mask_zero=True,
                                name='OneHot_smile')(input_)

        rnn_encoded = Bidirectional(LSTM(400, return_sequences=True),
                                    name='bidirectional_smile',
                                    merge_mode='sum',
                                    trainable=True)(input_embed)

        y_hat = AttentionDecoder(units =400,
                                name='attention_decoder_smile',
                                output_dim=2,
                                return_sequence=True,
                                return_probabilities=True,
                                return_attention=True,
                                trainable=True)(rnn_encoded)
        return Model(inputs = input_, outputs = y_hat)

    def run(self):
        self.model.compile(loss=self.loss_function,
                      optimizer=self.opt,
                      metrics=['acc'])
        return self.model.fit(self.X_train, 
                    self.Y_train, 
                    validation_data=(self.X_test, self.Y_test), 
                    epochs=self.epochs, 
                    batch_size=self.batch_size,
                    callbacks = [self.early,self.metrics,self.tensorBoard])
    