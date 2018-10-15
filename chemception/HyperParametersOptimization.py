import pixiedust
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
%load_ext autoreload
%autoreload 2
import input as data
import numpy as np

import talos as ta
from keras.optimizers import Adam,RMSprop

from keras.models import Sequential, Model
from keras.layers import Dropout, Dense, Input
from keras.optimizers import Adam, Nadam
from keras.activations import relu, elu, sigmoid
from keras.losses import binary_crossentropy


#Setting seed to re run the same simulation with the same result
seed = 7
nu.random.seed(seed)
model = ""

#Defining the size of the network, this can be passed as parameter
N=16
inputSize = 80
#This define which network we want to use:
#    C- Chemception
#    S- SMIELE to Vect
type = 'H'
executionName = "try"

main_execution_path = './build/'+executionName+'/'
final_resume         = main_execution_path + '_resume.txt'
# The data, split between train and test sets:
(XC, YC)     = data.LoadImageData(extensionImg='png',size=inputSize,duplicateProb=0,seed=seed)
(XV, YV,vocab,max_size)     = data.LoadSMILESData(duplicateProb=0,seed=seed)
#vocab = {'c': 1, '(': 2, ')': 3, 'o': 4, '=': 5, '1': 6, 'n': 7, '2': 8, '3': 9, '[': 10, ']': 11, 'f': 12, '4': 13, 's': 14, 'l': 15, '@': 16, 'h': 17, '5': 18, '+': 19, '-': 20, 'b': 21, 'r': 22, '\\': 23, '#': 24, '6': 25, '.': 26, '/': 27, 'i': 28, 'p': 29, '7': 30, '8': 31, 'a': 32, '%': 33, '9': 34, '0': 35, 'k': 36, 'e': 37, 'g': 38, 'm': 39, 't': 40, 'd': 41, 'v': 42, 'z': 43}
print(vocab)
vocab_size = len(vocab)
print(vocab_size)

cvscores = []
i=2

#K.clear_session()
model_name                          = type+'_trained_cross_'+str(i)
current_path                      = main_execution_path+model_name

model_name_file                  = model_name + '_model.h5'
model_directory                     = current_path+'/model'

model_path                          = os.path.join(model_directory, model_name_file)
log_dir                             = './build/logs/{}'.format(model_name)
resume_file                         = current_path + '/'+model_name+'_resume.txt'

X_trainC, X_testC, Y_trainC, Y_testC = train_test_split(XC, YC, test_size=0.1*i, random_state=seed)
X_trainV, X_testV, Y_trainV, Y_testV = train_test_split(XV, YV, test_size=0.1*i, random_state=seed)


# Convert class vectors to binary class matrices.
Y_trainC             = keras.utils.to_categorical(Y_trainC, 2)
Y_testC                 = keras.utils.to_categorical(Y_testC, 2)
Y_trainV             = keras.utils.to_categorical(Y_trainV, 2)
Y_testV                 = keras.utils.to_categorical(Y_testV, 2)

metrics = Metrics()

import numpy as np
import cv2
from keras.utils import to_categorical
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
from keras.models import Model
import keras
from network.layers import AttentionDecoder
from keras.utils import plot_model

from network.optimizer import Optimizer

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
import keras.backend as K

def TestChemception(x_train, y_train, x_test, y_test, params):
    input_img = Input(shape = (80, 80, 3),name='image_input')
    #Stem
    stem = Conv2D(n,(4,4),strides=2,name='Stem_Conv_2D',padding='same',activation='relu')(input_img)
    #IncResA
    inputs = keras.layers.Activation('relu',name='IncResNetA_activation_1')(stem)
    #First internal layer
    con = Conv2D(n,(1,1),strides=1, padding='same',name='IncResNetA_Conv2D_1',activation='relu')(inputs)
    conv = Conv2D(n,(1,1),strides=1, padding='same',name='IncResNetA_Conv2D_2',activation='relu')(inputs)
    Conv = Conv2D(n,(1,1),strides=1, padding='same',name='IncResNetA_Conv2D_3',activation='relu')(inputs)

    #second internal layer    
    conv3 = Conv2D(n, (3,3),strides=1,name='IncResNetA_Conv2D_4', padding='same')(conv)
    Conv3  = Conv2D(int(1.5*n), (3,3),strides=1,name='IncResNetA_Conv2D_5', padding='same')(Conv)

    #third internal layer
    Conv33  = Conv2D(int(2*n), (3,3),strides=1,name='IncResNetA_Conv2D_6', padding='same')(Conv3)

    concat = keras.layers.Concatenate(axis=-1,name='IncResNetA_Concat')([con,conv3,Conv33])
    convInc = Conv2D(n, (1,1),strides=1, padding='same',name='IncResNetA_Conv2D_7',activation='linear')(concat)

    IncResNetA = keras.layers.Add(name='IncResNetA_Add')([input, convInc])
    activation = keras.layers.Activation('relu',name='IncResNetA_activation_2')(IncResNetA)
    
    #RedA
    name = ''
    pool = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid',name='ReductionA_pool_1'+name)(activation)
    Conv = Conv2D(int(1.5*n),(3,3),strides=2, padding='valid',activation='relu',name='ReductionA_Conv2D_1'+name)(activation)
    conv = Conv2D(n,(1,1),strides=1, padding='same',activation='relu',name='ReductionA_Conv2D_2'+name)(activation)

    conv3 = Conv2D(n,(3,3),strides=1, padding='same',activation='relu',name='ReductionA_Conv2D_3'+name)(conv)
    conv33 = Conv2D(int(1.5*n),(3,3),strides=2, padding='valid',activation='relu',name='ReductionA_Conv2D_4'+name)(conv3)

    concat = keras.layers.Concatenate(axis=-1,name='ReductionA_concat'+name)([conv33,pool,Conv])
    activation = keras.layers.Activation('relu',name='ReductionA_activation_1'+name)(concat)
    
    #IncResB
    inputs = keras.layers.Activation('relu',name='IncResNetB_activation_1')(activation)
    #First internal layer
    con = Conv2D(n,(1,1),strides=1, padding='same',activation='relu',name='IncResNetB_Conv2D_1')(inputs)
    Conv = Conv2D(n,(1,1),strides=1, padding='same',activation='relu',name='IncResNetB_Conv2D_2')(inputs)

    #second internal layer    
    Conv3  = Conv2D(int(1.25*n), (1,7),strides=1, padding='same',name='IncResNetB_Conv2D_3')(Conv)
    #third internal layer
    Conv33  = Conv2D(int(1.5*n), (7,1),strides=1, padding='same',name='IncResNetB_Conv2D_4')(Conv3)

    concat = keras.layers.Concatenate(axis=-1,name='IncResNetB_Concat')([con,Conv33])
    convInc = Conv2D(n*4, (1,1),strides=1, padding='same',activation='linear',name='IncResNetB_Conv2D_6')(concat)

    IncResNetB = keras.layers.Add(name='IncResNetB_Add')([input, convInc])
    activation = keras.layers.Activation('relu',name='IncResNetB_activation_2')(IncResNetB)
    
    #RedB
    name='_2'
    pool = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid',name='ReductionA_pool_1'+name)(activation)
    Conv = Conv2D(int(1.5*n),(3,3),strides=2, padding='valid',activation='relu',name='ReductionA_Conv2D_1'+name)(activation)
    conv = Conv2D(n,(1,1),strides=1, padding='same',activation='relu',name='ReductionA_Conv2D_2'+name)(activation)

    conv3 = Conv2D(n,(3,3),strides=1, padding='same',activation='relu',name='ReductionA_Conv2D_3'+name)(conv)
    conv33 = Conv2D(int(1.5*n),(3,3),strides=2, padding='valid',activation='relu',name='ReductionA_Conv2D_4'+name)(conv3)

    concat = keras.layers.Concatenate(axis=-1,name='ReductionA_concat'+name)([conv33,pool,Conv])
    activation = keras.layers.Activation('relu',name='ReductionA_activation_1'+name)(concat)
    
    #IncResC
     #First internal layer
    inputs = keras.layers.Activation('relu',name='IncResNetC_activation_1')(activation)
    con = Conv2D(n,(1,1),strides=1, padding='same',activation='relu',name='IncResNetC_Conv2D_1')(inputs)
    Conv = Conv2D(n,(1,1),strides=1, padding='same',activation='relu',name='IncResNetC_Conv2D_2')(inputs)

    #second internal layer    
    Conv3  = Conv2D(int(1.16*n), (1,3),strides=1, padding='same',name='IncResNetC_Conv2D_3')(Conv)
    #third internal layer
    Conv33  = Conv2D(int(1.33*n), (3,1),strides=1, padding='same',name='IncResNetC_Conv2D_4')(Conv3)

    concat = keras.layers.Concatenate(axis=-1,name='IncResNetC_Concat')([con,Conv33])
    convInc = Conv2D(n*7, (1,1),strides=1, padding='same',activation='linear',name='IncResNetC_Conv2D_6')(concat)

    IncResNetC = keras.layers.Add(name='IncResNetC_Add')([input, convInc])
    activation = keras.layers.Activation('relu',name='IncResNetC_activation_2')(IncResNetC)
    
    #Pooling
    pool     = keras.layers.GlobalAveragePooling2D()(activation)
    out        = Dense(2, activation='linear')(pool)
    model = Model(inputs = input_img, outputs = out)
    
    model.compile(loss=self.loss_function,
                    optimizer=opt,
                    metrics=['accuracy'])
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
    history = model.fit_generator(datagen.flow(x_train, y_train,batch_size=self.batch_size),
        epochs=self.epochs/2,
        workers=4,
        validation_data=(x_test,y_test))
    return history,model

from network.layers import AttentionDecoder
def TestAttention(x_train, y_train, x_val, y_val, params):
        vocab_size = 53
        max_length = 498


        input_ = Input(shape=(max_length,), dtype='float32',name='text_input')
        input_embed = Embedding(vocab_size+1, params['first_neuron'],
                                input_length=max_length,
                                trainable=True,
                                mask_zero=True,
                                name='OneHot_smile')(input_)

        rnn_encoded = Bidirectional(params['cell'](params['first_neuron'], return_sequences=True),
                                    name='bidirectional_smile',
                                    merge_mode='concat',
                                    trainable=True)(input_embed)

        y_hat = AttentionDecoder(units =params['first_neuron'],
                                name='attention_decoder_smile',
                                output_dim=2,
                                return_sequence=True,
                                return_probabilities=False,
                                trainable=True)(rnn_encoded)
        drop = Dropout(params['dropout'])(y_hat)
        dense = Dense(2, activation=params['last_activation'],name ='dense_smile')(drop)
        model = Model(inputs = input_, outputs = dense)

        print(model.summary())
        model.compile(loss=params['losses'],
                      optimizer=params['optimizer'](lr=params['lr']),
                      metrics=['acc'])
        history = model.fit(x_train, 
                    y_train, 
                    validation_data=(x_val, y_val), 
                    epochs=params['epochs'], 
                    batch_size=params['batch_size'],
                           verbose=1)
        return history,model

p = {'lr': ( 0.001, 0.05,0.5, 1),
     'first_neuron':[20, 100,400],
     'batch_size': (32, 64,128),
     'epochs': [100,200],
     'dropout': (0.2, 0.15,0.1),
     'weight_regulizer':[None],
     'emb_output_dims': [None],
     'optimizer': [Adam, RMSprop],
     'losses': ['logcosh','binary_crossentropy', 'mean_squared_error'],
     'activation':[relu, elu],
     'cell':[LSTM,GRU],
     'bidirectional':['sum', 'mul', 'concat'],
     'last_activation': ['sigmoid']}


t = ta.Scan(x=np.concatenate((X_trainV,X_testV)),
            y=np.concatenate((Y_trainV,Y_testV)),
            model=TestAttention,
            params=p,
            functional_model=True)
# use Scan object as input
r = ta.Reporting(h)

# use filename as input
r = ta.Reporting('att.csv')