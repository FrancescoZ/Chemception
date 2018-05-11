# import csv
# import time
# from time import sleep

# import constant
# import helpers
# import input
# from models import Compound

# from rdkit.Chem import AllChem
# from rdkit.Chem import Draw

# import numpy as np

# def lr_schedule(epoch):
#     """Learning Rate Schedule
#     Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
#     Called automatically every epoch as part of callbacks during training.
#     # Arguments
#         epoch (int): The number of epochs
#     # Returns
#         lr (float32): learning rate
#     """
#     lr = 1e-3
#     if epoch > 180:
#         lr *= 0.5e-3
#     elif epoch > 160:
#         lr *= 1e-3
#     elif epoch > 120:
#         lr *= 1e-2
#     elif epoch > 80:
#         lr *= 1e-1
#     print('Learning rate: ', lr)
#     return lr

# def resnet_layer(inputs,
#                  num_filters=16,
#                  kernel_size=3,
#                  strides=1,
#                  activation='relu',
#                  batch_normalization=True,
#                  conv_first=True):
#     """2D Convolution-Batch Normalization-Activation stack builder
#     # Arguments
#         inputs (tensor): input tensor from input image or previous layer
#         num_filters (int): Conv2D number of filters
#         kernel_size (int): Conv2D square kernel dimensions
#         strides (int): Conv2D square stride dimensions
#         activation (string): activation name
#         batch_normalization (bool): whether to include batch normalization
#         conv_first (bool): conv-bn-activation (True) or
#             activation-bn-conv (False)
#     # Returns
#         x (tensor): tensor as input to the next layer
#     """
#     conv = Conv2D(num_filters,
#                   kernel_size=kernel_size,
#                   strides=strides,
#                   padding='same',
#                   kernel_initializer='he_normal',
#                   kernel_regularizer=l2(1e-4))

#     x = inputs
#     if conv_first:
#         x = conv(x)
#         if batch_normalization:
#             x = BatchNormalization()(x)
#         if activation is not None:
#             x = Activation(activation)(x)
#     else:
#         if batch_normalization:
#             x = BatchNormalization()(x)
#         if activation is not None:
#             x = Activation(activation)(x)
#         x = conv(x)
#     return x


# def resnet_v1(input_shape, depth, num_classes=10):
#     """ResNet Version 1 Model builder [a]
#     Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
#     Last ReLU is after the shortcut connection.
#     At the beginning of each stage, the feature map size is halved (downsampled)
#     by a convolutional layer with strides=2, while the number of filters is
#     doubled. Within each stage, the layers have the same number filters and the
#     same number of filters.
#     Features maps sizes:
#     stage 0: 32x32, 16
#     stage 1: 16x16, 32
#     stage 2:  8x8,  64
#     The Number of parameters is approx the same as Table 6 of [a]:
#     ResNet20 0.27M
#     ResNet32 0.46M
#     ResNet44 0.66M
#     ResNet56 0.85M
#     ResNet110 1.7M
#     # Arguments
#         input_shape (tensor): shape of input image tensor
#         depth (int): number of core convolutional layers
#         num_classes (int): number of classes (CIFAR10 has 10)
#     # Returns
#         model (Model): Keras model instance
#     """
#     if (depth - 2) % 6 != 0:
#         raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
#     # Start model definition.
#     num_filters = 16
#     num_res_blocks = int((depth - 2) / 6)

#     inputs = Input(shape=input_shape)
#     x = resnet_layer(inputs=inputs)
#     # Instantiate the stack of residual units
#     for stack in range(3):
#         for res_block in range(num_res_blocks):
#             strides = 1
#             if stack > 0 and res_block == 0:  # first layer but not first stack
#                 strides = 2  # downsample
#             y = resnet_layer(inputs=x,
#                              num_filters=num_filters,
#                              strides=strides)
#             y = resnet_layer(inputs=y,
#                              num_filters=num_filters,
#                              activation=None)
#             if stack > 0 and res_block == 0:  # first layer but not first stack
#                 # linear projection residual shortcut connection to match
#                 # changed dims
#                 x = resnet_layer(inputs=x,
#                                  num_filters=num_filters,
#                                  kernel_size=1,
#                                  strides=strides,
#                                  activation=None,
#                                  batch_normalization=False)
#             x = keras.layers.add([x, y])
#             x = Activation('relu')(x)
#         num_filters *= 2

#     # Add classifier on top.
#     # v1 does not use BN after last shortcut connection-ReLU
#     x = AveragePooling2D(pool_size=8)(x)
#     y = Flatten()(x)
#     outputs = Dense(num_classes,
#                     activation='softmax',
#                     kernel_initializer='he_normal')(y)

#     # Instantiate model.
#     model = Model(inputs=inputs, outputs=outputs)
#     return model


# def resnet_v2(input_shape, depth, num_classes=10):
#     """ResNet Version 2 Model builder [b]
#     Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
#     bottleneck layer
#     First shortcut connection per layer is 1 x 1 Conv2D.
#     Second and onwards shortcut connection is identity.
#     At the beginning of each stage, the feature map size is halved (downsampled)
#     by a convolutional layer with strides=2, while the number of filter maps is
#     doubled. Within each stage, the layers have the same number filters and the
#     same filter map sizes.
#     Features maps sizes:
#     conv1  : 32x32,  16
#     stage 0: 32x32,  64
#     stage 1: 16x16, 128
#     stage 2:  8x8,  256
#     # Arguments
#         input_shape (tensor): shape of input image tensor
#         depth (int): number of core convolutional layers
#         num_classes (int): number of classes (CIFAR10 has 10)
#     # Returns
#         model (Model): Keras model instance
#     """
#     if (depth - 2) % 9 != 0:
#         raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
#     # Start model definition.
#     num_filters_in = 16
#     num_res_blocks = int((depth - 2) / 9)

#     inputs = Input(shape=input_shape)
#     # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
#     x = resnet_layer(inputs=inputs,
#                      num_filters=num_filters_in,
#                      conv_first=True)

#     # Instantiate the stack of residual units
#     for stage in range(3):
#         for res_block in range(num_res_blocks):
#             activation = 'relu'
#             batch_normalization = True
#             strides = 1
#             if stage == 0:
#                 num_filters_out = num_filters_in * 4
#                 if res_block == 0:  # first layer and first stage
#                     activation = None
#                     batch_normalization = False
#             else:
#                 num_filters_out = num_filters_in * 2
#                 if res_block == 0:  # first layer but not first stage
#                     strides = 2    # downsample

#             # bottleneck residual unit
#             y = resnet_layer(inputs=x,
#                              num_filters=num_filters_in,
#                              kernel_size=1,
#                              strides=strides,
#                              activation=activation,
#                              batch_normalization=batch_normalization,
#                              conv_first=False)
#             y = resnet_layer(inputs=y,
#                              num_filters=num_filters_in,
#                              conv_first=False)
#             y = resnet_layer(inputs=y,
#                              num_filters=num_filters_out,
#                              kernel_size=1,
#                              conv_first=False)
#             if res_block == 0:
#                 # linear projection residual shortcut connection to match
#                 # changed dims
#                 x = resnet_layer(inputs=x,
#                                  num_filters=num_filters_out,
#                                  kernel_size=1,
#                                  strides=strides,
#                                  activation=None,
#                                  batch_normalization=False)
#             x = keras.layers.add([x, y])

#         num_filters_in = num_filters_out

#     # Add classifier on top.
#     # v2 has BN-ReLU before Pooling
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = AveragePooling2D(pool_size=8)(x)
#     y = Flatten()(x)
#     outputs = Dense(num_classes,
#                     activation='softmax',
#                     kernel_initializer='he_normal')(y)

#     # Instantiate model.
#     model = Model(inputs=inputs, outputs=outputs)
#     return model




# from keras.utils import to_categorical
# Y_train = to_categorical(Y_train)
# X_train = X_train / 255.0

# from keras.layers import Input
# from keras.layers import Conv2D, MaxPooling2D
# from keras.layers import Flatten, Dense
# from keras.models import Model
# from keras.optimizers import SGD
# import keras

# input_img = Input(shape = (300, 300, 3))
# model = Model(inputs = input_img, outputs = out)
# out    = Dense(2, activation='linear')(output)
# #IncResNet A
# stem = Conv2D(900,(4,4),stride=2, padding='valid',activation='relu')(input_img)
# model.add(stem)

# conv1 = Conv2D(900, (1,1),stride=1, padding='same')(stem)

# conv13 = Conv2D(900, (1,1),stride=1, padding='same')(stem)
# conv3 = Conv2D(900, (3,3),stride=1, padding='same')(conv13)

# conv133 = Conv2D(900, (1,1),stride=1, padding='same')(stem)
# conv33 = Conv2D(900, (3,3),stride=1, padding='same')(conv133)
# conv3fin = Conv2D(900, (3,3),stride=1, padding='same')(conv33)

# incConv = keras.layers.Concatenate(axis=-1)([conv1,conv3,conv3fin])

# convInc = Conv2D(900, (1,1),stride=1, padding='same')(incConv)

# IncResNetA = keras.layers.Add()([stem, convInc])
# activation = keras.layers.Activation('relu')(IncResNetA)

# #Reduction A
# conv21 = Conv2D(900, (1,1),stride=1, padding='same')(activation)
# conv233 = Conv2D(900, (3,3),stride=1, padding='same')(conv21)
# conv2333 = Conv2D(900, (3,3),stride=2, padding='same')(conv233)

# conv23 = Conv2D(900, (3,3),stride=2, padding='same')(activation)

# pool = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(activation)

# redConv = keras.layers.Concatenate(axis=-1)([pool,conv23,conv2333])
# actRed = keras.layers.Activation('relu')(redConv)

# #InResNet B

# tower_1 = Conv2D(900, (3,3), padding='same', activation='relu')(tower_1)

# tower_2 = Conv2D(900, (1,1), padding='same', activation='relu')(input_img)
# tower_2 = Conv2D(900, (5,5), padding='same', activation='relu')(tower_2)

# tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
# tower_3 = Conv2D(900, (1,1), padding='same', activation='relu')(tower_3)

# output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis = 3)
# output = Flatten()(output)
# out    = Dense(2, activation='softmax')(output)


# print(model.summary())

# epochs = 25
# lrate = 0.01
# decay = lrate/epochs
# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# model.fit(X_train, Y_train, validation_data=(X_train, Y_train), epochs=epochs, batch_size=32)

# from keras.models import model_from_json
# import os

# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# model.save_weights(os.path.join(os.getcwd(), 'model.h5'))
