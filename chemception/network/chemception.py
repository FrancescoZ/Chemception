import numpy as np
from keras.utils import to_categorical
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
from keras.models import Model
import keras
from keras.utils import plot_model

from network.optimizer import Optimizer

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
import keras.backend as K

class Chemception:
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

	def __init__(self,
					n,
					inputSize, 
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
					data_augmentation,
					metrics,
					tensorBoard,
					early):
		input_img = Input(shape = (inputSize, inputSize, 3))
		stem	= Chemception.Stem(input_img,n)
		incResA = Chemception.IncResNetA(stem,n)
		redA 	= Chemception.ReductionA(incResA,n)
		incResB = Chemception.IncResNetB(redA,n)
		redB 	= Chemception.ReductionA(incResB,n)
		incResC = Chemception.IncResNetC(redB,n)
		
		pool 	= keras.layers.GlobalAveragePooling2D()(incResC)
		out    	= Dense(2, activation='linear')(pool)

		self.model = Model(inputs = input_img, outputs = out)
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

		self.data_augmentation = data_augmentation

		self.metrics = metrics
		self.tensorBoard = tensorBoard
		self.early = early
		print(self.model.summary())

	
	def print(self):
		plot_model(self.model, to_file='modelChemception.png')
	
	def run(self):
		x_train 			= self.X_train
		y_train 			= self.Y_train
		X_test				= self.X_test
		Y_test 				= self.Y_test

		print('x_train shape:', x_train.shape)
		print(x_train.shape[0], 'train samples')

		x_train 			= self.X_train.astype('float32')
		X_test 				= self.X_test.astype('float32')

		x_train 			/= 255
		X_test 				/= 255
		

		# initiate RMSprop optimizer
		opt 				= keras.optimizers.RMSprop(lr=self.learning_rate, rho=self.rho, epsilon=self.epsilon, decay=0.0)

		# Let's train the model using RMSprop
		self.model.compile(loss=self.loss_function,
					optimizer=opt,
					metrics=['accuracy'])
		learning_rate_init	= 1e-3
		momentum			= 0.9
		gamma				= 0.92
		sgd = SGD(lr=learning_rate_init, decay=0, momentum=momentum, nesterov=True)
		optCallback = Optimizer.OptimizerTracker()
		

		if not self.data_augmentation:
			print('Not using data augmentation.')
			datagen = ImageDataGenerator(
				featurewise_center=False,  # set input mean to 0 over the dataset
				samplewise_center=False,  # set each sample mean to 0
				featurewise_std_normalization=False,  # divide inputs by std of the dataset
				samplewise_std_normalization=False,  # divide each input by its std
				zca_whitening=False,  # apply ZCA whitening
				rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
				width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
				height_shift_range=0,  # randomly shift images vertically (fraction of total height)
				horizontal_flip=False,  # randomly flip images
				vertical_flip=False)  # randomly flip images

			# Compute quantities required for feature-wise normalization
			# (std, mean, and principal components if ZCA whitening is applied).
			datagen.fit(x_train)
			# Fit the model on the batches generated by datagen.flow().
			self.model.fit_generator(datagen.flow(x_train, y_train,batch_size=self.batch_size),
				epochs=self.epochs/2,
				workers=4,
				validation_data=(X_test,Y_test),
				callbacks = [self.tensorBoard,self.metrics])
			model.fit_generator(datagen.flow(x_train, y_train,
				batch_size=self.batch_size),
				epochs=self.epochs/2,
				workers=4,
				validation_data=(X_test,Y_test),
				callbacks = [self.tensorBoard, optCallback,self.metrics,self.early])
		else:
			print('Using real-time data augmentation.')
			# This will do preprocessing and realtime data augmentation:
			datagen = ImageDataGenerator(
				featurewise_center=False,  # set input mean to 0 over the dataset
				samplewise_center=False,  # set each sample mean to 0
				featurewise_std_normalization=False,  # divide inputs by std of the dataset
				samplewise_std_normalization=False,  # divide each input by its std
				zca_whitening=False,  # apply ZCA whitening
				rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
				width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
				height_shift_range=0,  # randomly shift images vertically (fraction of total height)
				horizontal_flip=True,  # randomly flip images
				vertical_flip=True)  # randomly flip images

			# Compute quantities required for feature-wise normalization
			# (std, mean, and principal components if ZCA whitening is applied).
			datagen.fit(x_train)
			# Fit the model on the batches generated by datagen.flow().
			model.fit_generator(datagen.flow(x_train, y_train,batch_size=self.batch_size),
				epochs=self.epochs/2,
				workers=4,
				validation_data=(X_test,Y_test),
				callbacks = [self.tensorBoard,self.metrics])
			model.fit_generator(datagen.flow(x_train, y_train,
				batch_size=self.batch_size),
				epochs=self.epochs/2,
				workers=4,
				validation_data=(X_test,Y_test),
				callbacks = [self.tensorBoard, optCallback,self.metrics])