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
from network import Chemception
from network import SMILE2Vector 
from network import HATT
import input as data
import helpers
import Optimizer
from evaluation import Metrics
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

class HATT:
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
				tensorBoard):
		self.model = Sequential()
		self.model.add(Embedding(vocab_size+1, 100, input_length=max_length,
							trainable = True,
							mask_zero=False))
		self.model.add(Bidirectional(GRU(100, return_sequences=True)))
		self.model.add(AttLayer(100))
		self.model.add(Dense(2, activation='softmax'))

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


		print(self.model.summary())

	def run(self):
		
		self.model.compile(loss=self.loss_function,
              		optimizer='rmsprop',
              		metrics=['acc'])
		return self.model.fit(self.X_train, 
					self.Y_train, 
					validation_data=(self.X_test, self.Y_test), 
					epochs=self.epochs, 
					batch_size=self.batch_size,
					callbacks = [self.tensorBoard,self.metrics])

class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

