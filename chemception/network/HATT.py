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
from keras.utils import plot_model

class HATT:
	def __init__(self,
				char_index,
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
		MAX_WORD_LENGTH = 180
		MAX_NB_CHARS = 180
		EMBEDDING_DIM = 180
		embedding_layer = Embedding(len(char_index) + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_WORD_LENGTH,
                            trainable=True)

		char_input = Input(shape=(MAX_WORD_LENGTH,), dtype='int32')
		char_sequences = embedding_layer(char_input)
		char_lstm = Bidirectional(GRU(100, return_sequences=True))(char_sequences)
		char_dense = TimeDistributed(Dense(200))(char_lstm)
		char_att = AttentionWithContext()(char_dense)
		charEncoder = Model(char_input, char_att)

		words_input = Input(shape=(MAX_WORD_LENGTH, MAX_WORD_LENGTH), dtype='int32')
		words_encoder = TimeDistributed(charEncoder)(words_input)
		words_lstm = Bidirectional(GRU(100, return_sequences=True))(words_encoder)
		words_dense = TimeDistributed(Dense(200))(words_lstm)
		words_att = AttentionWithContext()(words_dense)
		preds = Dense(2, activation='sigmoid')(words_att)
		self.model = Model(words_input, preds)



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

	def print(self):
		plot_model(self.model, to_file='modelHATT.png')

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
def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]