from tensorflow.keras.layers import BatchNormalization, Conv2D, Flatten, Dropout, Dense
from tensorflow.keras import models
from tensorflow.keras.regularizers import l2
from keras.utils import np_utils
import numpy as np
from .callback import TrainingCallback



class CNN():
    def __init__(self, model_path, class_names, input_shape):
        self.model_path = model_path
        self.class_names = class_names
        self.class_num = len(self.class_names)
        self.input_shape = input_shape
        self.model = None


    def load_model(self):
        try:
            self.model = models.load_model(self.model_path)
        except:
            print("There is no trained model in", self.model_path)
            
    
    def train_model(self, train_data, valid_data, epoch_num=40):
        self.model = self.build()
        self.model.summary()
        
        training = self.reshape_data(train_data)
        validation = self.reshape_data(valid_data)
        self.train(training, validation, epoch_num)
    
    
    def reshape_data(self, data):
        sample_num = len(data[0])
        x = np.reshape(data[0], (sample_num, *self.input_shape))
        y = np_utils.to_categorical(data[1], num_classes=self.class_num)
        return x,y
    
    
    def train(self, train_data, valid_data, epoch_num):
        tc = TrainingCallback(self.model_path, self.class_names, valid_data)
        self.model.fit(train_data[0], train_data[1], 
                       validation_data = valid_data,
                       epochs = epoch_num, 
                       callbacks = [tc])


    def build(self):
        kernel = (3,3)
        strid = (2, 2)
        pad = 'same'
        activ = 'relu'
        init = 'he_normal'
        reg = l2(0.0005)
        
        model = models.Sequential()
        model.add(Conv2D(16, kernel_size=kernel, strides=strid, padding=pad,
                         kernel_initializer=init, kernel_regularizer=reg, 
                         input_shape=self.input_shape))
        
        model.add(Conv2D(32, kernel_size=kernel, strides=strid, padding=pad,
                         kernel_initializer=init, kernel_regularizer=reg, 
                         activation=activ))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=kernel, strides=strid, padding=pad,
                         kernel_initializer=init, kernel_regularizer=reg,
                         activation=activ))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))     
        
        model.add(Conv2D(64, kernel_size=kernel, strides=strid, padding=pad,
                         kernel_initializer=init, kernel_regularizer=reg, 
                         activation=activ))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=kernel, strides=strid, padding=pad,
                         kernel_initializer=init, kernel_regularizer=reg, 
                         activation=activ))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))
        
        model.add(Conv2D(128, kernel_size=kernel, strides=strid, padding=pad,
                         kernel_initializer=init, kernel_regularizer=reg, 
                         activation=activ))
        model.add(BatchNormalization())        
        model.add(Conv2D(128, kernel_size=kernel, strides=strid, padding=pad,
                         kernel_initializer=init, kernel_regularizer=reg, 
                         activation=activ))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))
        
        # y-connected layer
        model.add(Flatten())
        model.add(Dense(512, kernel_initializer=init, activation=activ))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(self.class_num, activation='softmax'))
        
        model.compile(loss='categorical_crossentropy', 
                            optimizer="adam",
                            metrics=['accuracy'])
        return model


    def predict(self, data):
        x_data = np.reshape(data, (1, *self.input_shape))
        predict = self.model.predict(x_data)
        y_pred = np.argmax(predict, axis=-1)[0]
        y_pred_perc = np.round(np.squeeze(predict)[y_pred]*100,1)
        
        if y_pred_perc>50:
            print(f'{self.class_names[y_pred]} ({y_pred_perc})')
        else:
            print('signal ignored')