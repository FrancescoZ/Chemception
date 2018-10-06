from network import Chemception
from network import VisualATT
from keras.layer import Concatenate

class ToxNet:

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
                    early,
                    vocab_size,
                    max_size):
       (imageInput, chemception) = Chemception(n,
                                    inputSize,
                                    X_trainC,
                                    Y_trainC,
                                    X_testC,
                                    Y_testC,
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
                                    early, 
                                    True).Concat()
        (textInput,toxtext) = VisualATT( vocab_size,
                                            max_size,
                                            X_trainV,
                                            Y_trainV,
                                            X_testV,
                                            Y_testV,
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
                                            False).Concat()
        print(chemception.shape)
        mergedOut = keras.layers.concatenate([chemception,toxtext[0]])

        firstHidden = keras.layers.Dense(500,name='First_dense')(mergedOut)
        act = keras.layers.Activation('relu',name='First_act')(firstHidden)
        drop = keras.layers.Dropout(0.4,name='First_drop')(act)

        secondHidden = Dense(300,name='Second_dense')(drop)
        act = Activation('relu',name='Second_act')(secondHidden)
        drop = Dropout(0.4,name='Second_drop')(act)

        thirdHidden = Dense(2,name='Thrid_dense')(drop)
        act = Activation('softmax',name='Third_act')(thirdHidden)

        model = keras.models.Model(inputs = [imageInput,textInput], outputs = act)
        print(model.summary())
        keras.utils.plot_model(model, to_file='modelToxNet.png')
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
    
    def get_output_layer(self, model, layer_name):
        # get the symbolic outputs of each "key" layer (we gave them unique names).
        layer_dict = dict([(layer.name, layer) for layer in model.layers])
        layer = layer_dict[layer_name]
        return layer


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
        opt                 = keras.optimizers.RMSprop(lr=self.learning_rate, rho=self.rho, epsilon=self.epsilon, decay=0.0)

        # Let's train the model using RMSprop
        self.model.compile(loss=self.loss_function,
                    optimizer=opt,
                    metrics=['accuracy'])
        learning_rate_init    = 1e-3
        momentum            = 0.9
        gamma                = 0.92
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
            self.model.fit_generator(datagen.flow(x_train, y_train,
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
            self.model.fit_generator(datagen.flow(x_train, y_train,batch_size=self.batch_size),
                epochs=self.epochs/2,
                workers=4,
                validation_data=(X_test,Y_test),
                callbacks = [self.tensorBoard,self.metrics])
            self.model.fit_generator(datagen.flow(x_train, y_train,
                batch_size=self.batch_size),
                epochs=self.epochs/2,
                workers=4,
                validation_data=(X_test,Y_test),
                callbacks = [self.tensorBoard, optCallback,self.metrics])