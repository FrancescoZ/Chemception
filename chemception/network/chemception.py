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

class Chemception:
    def Stem(input,n):
        stem = Conv2D(n,(4,4),strides=2,name='Stem_Conv_2D',padding='same',activation='relu')(input)
        # stem = Conv2D(n,(3,4),strides=2,name='Stem_Conv_2D',padding='valid',activation='relu')(input)
        # conv3 = Conv2D(n, (3,3),strides=1,name='Stem_Conv3D', padding='valid')(stem)
        # conv33 = Conv2D(n, (3,3),strides=1,name='Stem_Conv33D', padding='same')(conv3)

        # pool = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid',name='Stem_pool_1')(conv33)
        # conv333 = Conv2D(int(n*1.5), (3,3),strides=2,name='Stem_Conv333D', padding='valid')(conv33)
        # concat = keras.layers.Concatenate(axis=-1,name='Stem_Concat')([pool,conv333])

        # conv1 = Conv2D(int(n*2), (3,3),strides=1,name='Stem_Conv1D', padding='valid')(concat)
        # conv31 = Conv2D(int(n*1.5), (3,3),strides=1,name='Stem_Conv31D', padding='same')(conv1)

        # conv11 = Conv2D(n, (1,1),strides=1,name='Stem_Conv11D', padding='same')(concat)
        # conv71 = Conv2D(n, (7,1),strides=1,name='Stem_Conv71D', padding='same')(conv11)
        # conv17 = Conv2D(n, (1,7),strides=1,name='Stem_Conv17D', padding='same')(conv71)
        # conv733 = Conv2D(int(n*1.5), (3,3),strides=1,name='Stem_Conv733D', padding='valid')(conv17)

        # concat = keras.layers.Concatenate(axis=-1,name='Stem_Concat_2')([conv31,conv733])

        # pool = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid',name='Stem_pool_2')(concat)
        # conv333 = Conv2D(int(n*1.5), (3,3),strides=2,name='Stem_Conv3poolD_2', padding='valid')(concat)
        # concat = keras.layers.Concatenate(axis=-1,name='Stem_Concat_3')([pool,conv333])
        return stem

    def IncResNetA(input,n):
        input = keras.layers.Activation('relu',name='IncResNetA_activation_1')(input)
        #First internal layer
        con = Conv2D(n,(1,1),strides=1, padding='same',name='IncResNetA_Conv2D_1',activation='relu')(input)
        conv = Conv2D(n,(1,1),strides=1, padding='same',name='IncResNetA_Conv2D_2',activation='relu')(input)
        Conv = Conv2D(n,(1,1),strides=1, padding='same',name='IncResNetA_Conv2D_3',activation='relu')(input)

        #second internal layer    
        conv3 = Conv2D(n, (3,3),strides=1,name='IncResNetA_Conv2D_4', padding='same')(conv)
        Conv3  = Conv2D(int(1.5*n), (3,3),strides=1,name='IncResNetA_Conv2D_5', padding='same')(Conv)

        #third internal layer
        Conv33  = Conv2D(int(2*n), (3,3),strides=1,name='IncResNetA_Conv2D_6', padding='same')(Conv3)

        concat = keras.layers.Concatenate(axis=-1,name='IncResNetA_Concat')([con,conv3,Conv33])
        convInc = Conv2D(n, (1,1),strides=1, padding='same',name='IncResNetA_Conv2D_7',activation='linear')(concat)

        IncResNetA = keras.layers.Add(name='IncResNetA_Add')([input, convInc])
        activation = keras.layers.Activation('relu',name='IncResNetA_activation_2')(IncResNetA)

        return activation

    def IncResNetB(input,n):
        input = keras.layers.Activation('relu',name='IncResNetB_activation_1')(input)
        #First internal layer
        con = Conv2D(n,(1,1),strides=1, padding='same',activation='relu',name='IncResNetB_Conv2D_1')(input)
        Conv = Conv2D(n,(1,1),strides=1, padding='same',activation='relu',name='IncResNetB_Conv2D_2')(input)

        #second internal layer    
        Conv3  = Conv2D(int(1.25*n), (1,7),strides=1, padding='same',name='IncResNetB_Conv2D_3')(Conv)
        #third internal layer
        Conv33  = Conv2D(int(1.5*n), (7,1),strides=1, padding='same',name='IncResNetB_Conv2D_4')(Conv3)

        concat = keras.layers.Concatenate(axis=-1,name='IncResNetB_Concat')([con,Conv33])
        convInc = Conv2D(n*4, (1,1),strides=1, padding='same',activation='linear',name='IncResNetB_Conv2D_6')(concat)

        IncResNetB = keras.layers.Add(name='IncResNetB_Add')([input, convInc])
        activation = keras.layers.Activation('relu',name='IncResNetB_activation_2')(IncResNetB)

        return activation

    def IncResNetC(input,n):
        #First internal layer
        input = keras.layers.Activation('relu',name='IncResNetC_activation_1')(input)
        con = Conv2D(n,(1,1),strides=1, padding='same',activation='relu',name='IncResNetC_Conv2D_1')(input)
        Conv = Conv2D(n,(1,1),strides=1, padding='same',activation='relu',name='IncResNetC_Conv2D_2')(input)

        #second internal layer    
        Conv3  = Conv2D(int(1.16*n), (1,3),strides=1, padding='same',name='IncResNetC_Conv2D_3')(Conv)
        #third internal layer
        Conv33  = Conv2D(int(1.33*n), (3,1),strides=1, padding='same',name='IncResNetC_Conv2D_4')(Conv3)

        concat = keras.layers.Concatenate(axis=-1,name='IncResNetC_Concat')([con,Conv33])
        convInc = Conv2D(n*7, (1,1),strides=1, padding='same',activation='linear',name='IncResNetC_Conv2D_6')(concat)

        IncResNetC = keras.layers.Add(name='IncResNetC_Add')([input, convInc])
        activation = keras.layers.Activation('relu',name='IncResNetC_activation_2')(IncResNetC)

        return activation

    def ReductionA(input,n,name=''):
        pool = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid',name='ReductionA_pool_1'+name)(input)
        Conv = Conv2D(int(1.5*n),(3,3),strides=2, padding='valid',activation='relu',name='ReductionA_Conv2D_1'+name)(input)
        conv = Conv2D(n,(1,1),strides=1, padding='same',activation='relu',name='ReductionA_Conv2D_2'+name)(input)

        conv3 = Conv2D(n,(3,3),strides=1, padding='same',activation='relu',name='ReductionA_Conv2D_3'+name)(conv)
        conv33 = Conv2D(int(1.5*n),(3,3),strides=2, padding='valid',activation='relu',name='ReductionA_Conv2D_4'+name)(conv3)

        concat = keras.layers.Concatenate(axis=-1,name='ReductionA_concat'+name)([conv33,pool,Conv])
        activation = keras.layers.Activation('relu',name='ReductionA_activation_1'+name)(concat)

        return activation

    def ReductionB(input,n,name=''):
        pool = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid',name='ReductionB_pool_1'+name)(input)

        Conv = Conv2D(n,(1,1),strides=1, padding='same',activation='relu',name='ReductionB_Conv2D_1'+name)(input)
        conv = Conv2D(n,(1,1),strides=1, padding='same',activation='relu',name='ReductionB_Conv2D_2'+name)(input)
        CONV = Conv2D(n,(1,1),strides=1, padding='same',activation='relu',name='ReductionB_Conv2D_3'+name)(input)

        conv3 = Conv2D(int(1.5*n),(3,3),strides=2, padding='valid',activation='relu',name='ReductionB_Conv2D_4'+name)(conv)
        Conv3 = Conv2D(int(1.25*n),(3,3),strides=2, padding='valid',activation='relu',name='ReductionB_Conv2D_5'+name)(Conv)
        CONV3 = Conv2D(int(1.25*n),(3,1),strides=1, padding='same',activation='relu',name='ReductionB_Conv2D_6'+name)(CONV)

        CONV33 = Conv2D(int(1.25*n),(3,1),strides=2, padding='valid',activation='relu',name='ReductionB_Conv2D_7'+name)(CONV3)

        concat = keras.layers.Concatenate(axis=-1,name='ReductionB_concat'+name)([CONV33,pool,Conv3,conv3])
        activation = keras.layers.Activation('relu',name='ReductionB_activation_1'+name)(concat)

        return activation

    def __init__(self,
                    n,
                    inputSize, 
                    X_train,
                    Y_train,
                    X_test,
                    Y_test,
                    learning_rate,
                    optimizer,
                    epsilon,
                    epochs,
                    loss_function,
                    log_dir,
                    batch_size,
                    data_augmentation,
                    metrics,
                    tensorBoard,
                    early,
                    features,
                    classes = 2):
        # input_img = Input(shape = (inputSize, inputSize, 3),name='image_input')
        # stem    = Chemception.Stem(input_img,n)
        # incResA = Chemception.IncResNetA(stem,int(n*4.5))
        # redA     = Chemception.ReductionA(incResA,n)
        # incResB = Chemception.IncResNetB(redA,int(n*1.875))
        # redB     = Chemception.ReductionA(incResB,int(n),'_2')
        # incResC = Chemception.IncResNetC(redB,int(n*1.5))
        
        input_img = Input(shape = (inputSize, inputSize, 3),name='image_input')
        stem    = Chemception.Stem(input_img,n)
        incResA = Chemception.IncResNetA(stem,int(n))
        redA     = Chemception.ReductionA(incResA,n)
        incResB = Chemception.IncResNetB(redA,n)
        redB     = Chemception.ReductionA(incResB,n,'_2')
        incResC = Chemception.IncResNetC(redB,n)

        pool     = keras.layers.GlobalAveragePooling2D()(incResC)
        
        if not features:
            out        = Dense(2, activation='linear')(pool)
            self.model = Model(inputs = input_img, outputs = out)
        else:
            self.model = Model(inputs = input_img, outputs = pool)
        self.n = n
        self.inputSize = inputSize
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

        self.learning_rate = learning_rate
        #self.rho = rho
        self.epsilon = epsilon
        self.epochs = epochs

        self.loss_function = loss_function
        self.opt = optimizer
        self.log_dir = log_dir
        self.batch_size = batch_size

        self.data_augmentation = data_augmentation

        self.metrics = metrics
        self.tensorBoard = tensorBoard
        self.early = early
        self.input_img = input_img
        self.pool = pool
        print(self.model.summary())


    def get_output_layer(self, model, layer_name):
        # get the symbolic outputs of each "key" layer (we gave them unique names).
        layer_dict = dict([(layer.name, layer) for layer in model.layers])
        layer = layer_dict[layer_name]
        return layer
        
    def Concat(self):        
        return self.input_img, self.pool

    def Visualize(self,img_path, output_path):
        original_img = cv2.imread(img_path, 1)
        width, height, _ = original_img.shape
        #Reshape to the network input shape (3, w, h).
        #img = np.array([np.transpose(np.float32(original_img), (2, 0, 1))])

        #Get the 512 input weights to the softmax.
        class_weights = self.model.layers[-1].get_weights()[0]
        final_conv_layer = self.get_output_layer(self.model, "conv2d_26")
        get_output = K.function([self.model.layers[0].input], \
                    [final_conv_layer.output, 
                    self.model.layers[-1].output])
        [conv_outputs, predictions] = get_output([np.array([original_img])])
        conv_outputs = conv_outputs[0, :, :, :]
        print(predictions)
        #Create the class activation map.
        cam = np.ones(conv_outputs.shape[0 : 2], dtype = np.float32)
        target_class = 1

        for i, w in enumerate(class_weights[:, target_class]):
                cam+= w * conv_outputs[:, :,i]
                
        print("predictions", predictions)
        cam /= np.max(cam)
        cam = cv2.resize(cam, (height, width))
        print(cam.shape)
        cam /= np.max(cam)
        cam = cv2.resize(cam, (height, width))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.CV_8UC1)
        heatmap[np.where(cam < 0.2)] = 0
        img = heatmap*0.5 + original_img        
        cv2.imwrite(output_path, img)
    
    def printModel(self):
        plot_model(self.model, to_file='modelChemception.png')
    
    def run(self):
        x_train             = self.X_train
        y_train             = self.Y_train
        X_test                = self.X_test
        Y_test                 = self.Y_test

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')

        x_train             = self.X_train.astype('float32')
        X_test                 = self.X_test.astype('float32')

        x_train             /= 255
        X_test                 /= 255
        
        # initiate RMSprop optimizer
        #opt                 = keras.optimizers.RMSprop(lr=self.learning_rate, rho=self.rho, epsilon=self.epsilon, decay=0.0)

        # Let's train the model using RMSprop
        self.model.compile(loss=self.loss_function,
                    optimizer=self.opt,
                    metrics=['accuracy'])
        # learning_rate_init    = 1e-3
        # momentum            = 0.9
        # gamma                = 0.92
        # sgd = SGD(lr=learning_rate_init, decay=0, momentum=momentum, nesterov=True)
        # optCallback = Optimizer.OptimizerTracker()

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
                epochs=self.epochs,
                workers=4,
                steps_per_epoch=600,
                validation_data=(X_test,Y_test),
                callbacks = [self.tensorBoard,self.metrics])
            # self.model.fit_generator(datagen.flow(x_train, y_train,
            #     batch_size=self.batch_size),
            #     epochs=self.epochs/2,
            #     workers=4,
            #     steps_per_epoch=600,
            #     validation_data=(X_test,Y_test),
            #     callbacks = [self.tensorBoard, optCallback,self.metrics,self.early])
        # else:
        #     print('Using real-time data augmentation.')
        #     # This will do preprocessing and realtime data augmentation:
        #     datagen = ImageDataGenerator(
        #         featurewise_center=False,  # set input mean to 0 over the dataset
        #         samplewise_center=False,  # set each sample mean to 0
        #         featurewise_std_normalization=False,  # divide inputs by std of the dataset
        #         samplewise_std_normalization=False,  # divide each input by its std
        #         zca_whitening=False,  # apply ZCA whitening
        #         rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        #         width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        #         height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        #         horizontal_flip=True,  # randomly flip images
        #         vertical_flip=True)  # randomly flip images

        #     # Compute quantities required for feature-wise normalization
        #     # (std, mean, and principal components if ZCA whitening is applied).
        #     datagen.fit(x_train)
        #     # Fit the model on the batches generated by datagen.flow().
        #     self.model.fit_generator(datagen.flow(x_train, y_train,batch_size=self.batch_size),
        #         epochs=self.epochs/2,
        #         workers=1,
        #         validation_data=(X_test,Y_test),
        #         callbacks = [self.tensorBoard,self.metrics])
        #     self.model.fit_generator(datagen.flow(x_train, y_train,
        #         batch_size=self.batch_size),
        #         epochs=self.epochs/2,
        #         workers=1,
        #         validation_data=(X_test,Y_test),
        #         callbacks = [self.tensorBoard, optCallback,self.metrics])