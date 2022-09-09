from tensorflow.keras.layers import BatchNormalization, Conv2D, Flatten, Dropout, Dense
from tensorflow.keras import models
from tensorflow.keras.regularizers import l2
from keras.utils import np_utils
import numpy as np
import os
import tomli

import speechcommai.data.load as load
from .callback import TrainingCallback
# from record import AudioRecord

class AudioAI():
    def __init__(self, collection, train=False, epoch_num=40):        
        with open("config.toml", mode="rb") as fp:
            config = tomli.load(fp)

        #get essential data from the config file
        dirs = config['directories']
        audio = config['audio']
        model_dir = dirs['saved_models']
        self.prep_data_dir = dirs['preprocessed_data']
        self.class_names = config['data'][collection]
        
        
        rate = audio['rate']
        dur = audio['duration']
        hop = audio['hop_length']
        shape0 = audio['mfcc']
        shape1 = int(dur*rate/hop) + 1
        
        self.input_shape = (shape0, shape1, 1)
        
        self.model_subdir = os.path.join(model_dir, collection)
        self.class_num = len(self.class_names)


        if train:
            self.to_train(epoch_num)
        else:
            self.load_model(self.model_subdir)
        
    
    def load_model(self, model_path):
        try:
            self.model = models.load_model(model_path)
        except:
            print("There is no trained model in", model_path)
            self.to_train()
    
    
    def to_train(self, epoch_num):
        self.model = self.build()
        self.model.summary()

        validation, testing, training = load.load_data(self.prep_data_dir, self.class_names)
        self.train(epoch_num, self.prep_data(training), self.prep_data(validation))


    def prep_data(self, data):
        x = np.expand_dims(np.concatenate(data.mfcc), axis=-1)
        y = np_utils.to_categorical(np.concatenate(data.labels), num_classes=self.class_num)
        return x,y
    
    
    def train(self, epoch_num, train_data, valid_data):
        callback = TrainingCallback(self.model_subdir, self.class_names, valid_data)
        self.model.fit(train_data[0], train_data[1], 
                       validation_data = valid_data,
                       epochs = epoch_num, 
                       callbacks = [callback])


    def build(self):
        
        kernel = (3,3)
        strid = (2, 2)
        pad = "same"
        activ = 'relu'
        reg = l2(0.0005)
        
        model = models.Sequential()
        model.add(Conv2D(16, kernel_size=kernel, strides=strid, padding=pad,
                         kernel_regularizer=reg, input_shape=self.input_shape))
        model.add(Conv2D(16, kernel_size=kernel, strides=strid, padding=pad,
                         kernel_regularizer=reg, activation=activ))
        model.add(BatchNormalization())
        model.add(Conv2D(16, kernel_size=kernel, strides=strid, padding=pad,
                         kernel_regularizer=reg, activation=activ))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))     
        
        model.add(Conv2D(64, kernel_size=kernel, strides=strid, padding=pad,
                         kernel_regularizer=reg, activation=activ))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=kernel, strides=strid, padding=pad,
                         kernel_regularizer=reg, activation=activ))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))
        
        model.add(Conv2D(128, kernel_size=kernel, strides=strid, padding=pad,
                         kernel_regularizer=reg, activation=activ))
        model.add(BatchNormalization())        
        model.add(Conv2D(128, kernel_size=kernel, strides=strid, padding=pad,
                         kernel_regularizer=reg, activation=activ))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))
        
        # y-connected layer
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        model.add(Dense(self.class_num, activation='softmax'))
        
        model.compile(loss='categorical_crossentropy', 
                            optimizer="adam",
                            metrics=['accuracy'])
        return model


    def predict(self, data):
        x_data = np.expand_dims(data, axis=0)
        x_data = np.expand_dims(x_data, axis=-1)
        predict = self.model.predict(x_data)
        y_pred = np.argmax(predict, axis=-1)[0]
        y_pred_perc = np.round(np.squeeze(predict)[y_pred]*100,1)
        
        if y_pred_perc>50:
            print(f'{self.class_names[y_pred]} ({y_pred_perc})')
        else:
            print('signal ignored')
        


if __name__ == '__main__':
    data_collection = 'all'
    ai = AudioAI(data_collection, True)
    