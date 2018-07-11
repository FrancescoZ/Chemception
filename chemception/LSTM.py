
import numpy as nu
import network
import input as data
import helpers
import Optimizer
from evaluation import Metrics

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

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from sklearn.model_selection import train_test_split

# fix random seed for reproducibility
seed 				= 7
top_words 			= 24076
cross_val			= 3

nu.random.seed(seed)
# load the dataset but only keep the top n words, zero the rest
(X, Y) = data.LoadSMILESData()

cvscores = []
for i in range(2,cross_val):

	K.clear_session()
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1*i, random_state=seed)
	# create model	

	# truncate and pad input sequences
	max_review_length = 500
	X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
	X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

	# create the model
	embedding_vecor_length = 60
	model = Sequential()
	model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
	model.add(LSTM(384, return_sequences=True))
	model.add(LSTM(384))
	model.add(Dense(1, activation='softmax'))
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
	# create the model
	# embedding_vecor_length = 32
	# model = Sequential()
	# model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
	# model.add(LSTM(100))
	# model.add(Dense(1, activation='sigmoid'))
	# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# print(model.summary())
	# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)