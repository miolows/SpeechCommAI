from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras import models
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from keras.utils import np_utils
import numpy as np
import time
import os

import load
# from config import Configurator
from callbacks import TrainingCallback, PredictionCallback
# from record import AudioRecord

class AudioAI():
    def __init__(self, config, collection, train=False):
        self.config = config
        self.collection = collection
        
        #get essential data from config file
        model_dir = self.config.get('directories', 'Saved models')
        self.model_subdir = os.path.join(model_dir, collection)
        self.sample_shape = self.config.get('audio', 'sample shape')
        self.class_names = self.config.get('data', collection)
        self.class_num = len(self.class_names)
        
        if train:
            self.to_train()
        else:
            self.load_model(self.model_subdir)
        
    
    def load_model(self, model_path):
        try:
            self.model = models.load_model(model_path)
        except:
            print("There is no trained model in", model_path)
            self.to_train()
    
    
    def to_train(self):
        data_dir = self.config.get('directories', 'Preprocessed data')
        epoch_num = self.config.get('ai', 'epochs')
        self.model = self.build(l2(0.0005))
        self.model.summary()

        validation, testing, training = load.load_data(data_dir, self.class_names)
        
        self.train(epoch_num, self.prep_data(training), self.prep_data(validation))


    def prep_data(self, data):
        x = np.expand_dims(np.concatenate(data['mfcc']), axis=-1)
        y = np_utils.to_categorical(np.concatenate(data['labels']), num_classes=self.class_num)
        return x,y
    
    
    def train(self, epoch_num, train_data, valid_data):
        callback = TrainingCallback(self.model_subdir, self.class_names, valid_data)
        self.model.fit(train_data[0], train_data[1], 
                       validation_data = valid_data,
                       epochs = epoch_num, 
                       callbacks = [callback])


    def build(self, reg, init="he_normal"):
        # initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
        model = models.Sequential()
        inputShape = self.sample_shape
        chanDim = -1
# 		# if we are using "channels first", update the input shape
# 		# and channels dimension
#         if K.image_data_format() == "channels_first":
#             inputShape = (depth, height, width)
#             chanDim = 1
        
        model.add(Conv2D(16, (7, 7), strides=(2, 2), padding="valid",
			kernel_initializer=init, kernel_regularizer=reg,
			input_shape=inputShape))
		# here we stack two CONV layers on top of each other where
		# each layers will learn a total of 32 (3x3) filters
        model.add(Conv2D(32, (3, 3), padding="same",
                              kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="same",
                              kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))
        
        # stack two more CONV layers, keeping the size of each filter
		# as 3x3 but increasing to 64 total learned filters
        model.add(Conv2D(64, (3, 3), padding="same",
			kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same",
			kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))
		# increase the number of filters again, this time to 128
        model.add(Conv2D(128, (3, 3), padding="same",
			kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same",
			kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))
        
        # y-connected layer
        model.add(Flatten())
        model.add(Dense(512, kernel_initializer=init))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        #max classifier
        model.add(Dense(self.class_num))
        model.add(Activation("softmax"))
        model.compile(loss='categorical_crossentropy', 
                            optimizer="adam",
                            metrics=['accuracy'])
        return model

    def predict(self, data):
        x_data =  np.expand_dims(data, axis=0)
        x_data =  np.expand_dims(x_data, axis=-1)
        predict = self.model.predict(x_data)
        y_pred = np.argmax(predict, axis=-1)[0]
        y_pred_perc = np.squeeze(predict)[y_pred]
        # print(predict)
        # print(y_pred_perc)
        # list(map(lambda x: self.labels))
        if y_pred_perc<0.3:
            print("I don't know what you said. Maybe {}".format(self.class_names[y_pred]))
        elif y_pred_perc>0.3 and y_pred_perc<0.6: 
            print("I bet that you said {}".format(self.class_names[y_pred]))
        elif y_pred_perc>0.6 and y_pred_perc<0.8: 
            print("I'm pretty sure you said {}".format(self.class_names[y_pred]))
        else:
            print("You said {}".format(self.class_names[y_pred]))


          
if __name__ == '__main__':
    pass
    # config = Configurator()
    # data_collection = 'all'
    # ai = AudioAI(config, data_collection)

    # rec = AudioRecord(config)
    # for i in range(10):
        
    #     rec.record()
    #     audio_data = rec.get_live_rec_data()
    #     ai.predict(audio_data)
    #     time.sleep(3.0)