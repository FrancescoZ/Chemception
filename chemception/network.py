import numpy as np
from keras.utils import to_categorical
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.optimizers import SGD
import keras
from keras.utils import plot_model

def Stem(input,n):
	stem = Conv2D(n,(4,4),strides=2, padding='same',activation='relu')(input)
	return stem

def IncResNetA(input,n):
	input = keras.layers.Activation('relu')(input)
	#First internal layer
	con = Conv2D(n,(1,1),strides=1, padding='same',activation='relu')(input)
	conv = Conv2D(n,(1,1),strides=1, padding='same',activation='relu')(input)
	Conv = Conv2D(n,(1,1),strides=1, padding='same',activation='relu')(input)

	#second internal layer	
	conv3 = Conv2D(n, (3,3),strides=1, padding='same')(conv)
	Conv3  = Conv2D(int(1.5*n), (3,3),strides=1, padding='same')(Conv)

	#third internal layer
	Conv33  = Conv2D(int(2*n), (3,3),strides=1, padding='same')(Conv3)

	concat = keras.layers.Concatenate(axis=-1)([con,conv3,Conv33])
	convInc = Conv2D(n, (1,1),strides=1, padding='same',activation='linear')(concat)

	IncResNetA = keras.layers.Add()([input, convInc])
	activation = keras.layers.Activation('relu')(IncResNetA)

	return activation

def IncResNetB(input,n):
	input = keras.layers.Activation('relu')(input)
	#First internal layer
	con = Conv2D(n,(1,1),strides=1, padding='same',activation='relu')(input)
	Conv = Conv2D(n,(1,1),strides=1, padding='same',activation='relu')(input)

	#second internal layer	
	Conv3  = Conv2D(int(1.25*n), (1,7),strides=1, padding='same')(Conv)
	#third internal layer
	Conv33  = Conv2D(int(1.5*n), (7,1),strides=1, padding='same')(Conv3)

	concat = keras.layers.Concatenate(axis=-1)([con,Conv33])
	convInc = Conv2D(n*4, (1,1),strides=1, padding='same',activation='linear')(concat)

	IncResNetB = keras.layers.Add()([input, convInc])
	activation = keras.layers.Activation('relu')(IncResNetB)

	return activation

def IncResNetC(input,n):
	#First internal layer
	input = keras.layers.Activation('relu')(input)
	con = Conv2D(n,(1,1),strides=1, padding='same',activation='relu')(input)
	Conv = Conv2D(n,(1,1),strides=1, padding='same',activation='relu')(input)

	#second internal layer	
	Conv3  = Conv2D(int(1.16*n), (1,3),strides=1, padding='same')(Conv)
	#third internal layer
	Conv33  = Conv2D(int(1.33*n), (3,1),strides=1, padding='same')(Conv3)

	concat = keras.layers.Concatenate(axis=-1)([con,Conv33])
	convInc = Conv2D(n*7, (1,1),strides=1, padding='same',activation='linear')(concat)

	IncResNetC = keras.layers.Add()([input, convInc])
	activation = keras.layers.Activation('relu')(IncResNetC)

	return activation

def ReductionA(input,n):
	pool = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(input)
	Conv = Conv2D(int(1.5*n),(3,3),strides=2, padding='valid',activation='relu')(input)
	conv = Conv2D(n,(1,1),strides=1, padding='same',activation='relu')(input)

	conv3 = Conv2D(n,(3,3),strides=1, padding='same',activation='relu')(conv)
	conv33 = Conv2D(int(1.5*n),(3,3),strides=2, padding='valid',activation='relu')(conv3)

	concat = keras.layers.Concatenate(axis=-1)([conv33,pool,Conv])
	activation = keras.layers.Activation('relu')(concat)

	return activation

def ReductionB(input,n):
	pool = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(input)

	Conv = Conv2D(n,(1,1),strides=1, padding='same',activation='relu')(input)
	conv = Conv2D(n,(1,1),strides=1, padding='same',activation='relu')(input)
	CONV = Conv2D(n,(1,1),strides=1, padding='same',activation='relu')(input)

	conv3 = Conv2D(int(1.5*n),(3,3),strides=2, padding='valid',activation='relu')(conv)
	Conv3 = Conv2D(int(1.25*n),(3,3),strides=2, padding='valid',activation='relu')(Conv)
	CONV3 = Conv2D(int(1.25*n),(3,1),strides=1, padding='same',activation='relu')(CONV)

	CONV33 = Conv2D(int(1.25*n),(3,1),strides=2, padding='valid',activation='relu')(CONV3)

	concat = keras.layers.Concatenate(axis=-1)([CONV33,pool,Conv3,conv3])
	activation = keras.layers.Activation('relu')(concat)

	return activation

def Chemception(n,inputSize):
	input_img = Input(shape = (inputSize, inputSize, 3))
	stem	= Stem(input_img,n)
	incResA = IncResNetA(stem,n)
	redA 	= ReductionA(incResA,n)
	incResB = IncResNetB(redA,n)
	redB 	= ReductionA(incResB,n)
	incResC = IncResNetC(redB,n)
	
	pool 	= keras.layers.GlobalAveragePooling2D()(incResC)
	out    	= Dense(2, activation='linear')(pool)

	model = Model(inputs = input_img, outputs = out)

	print(model.summary())
	#plot_model(model, to_file='model.png')
	return model