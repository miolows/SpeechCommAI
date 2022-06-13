from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras.models import Sequential, load_model 
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from keras.utils import np_utils
import numpy as np
import time
import os

from prep import load_data
from callbacks import TrainingCallback, PredictionCallback
from record import AudioRecord

class AudioAI():
    def __init__(self, num_classes, models_dir, to_train=False):
        self.models_dir = models_dir
        self.num_classes = num_classes
        if to_train:
            self.to_train(num_classes, 40)
            
        else:
            model_subdir = os.path.join(models_dir, "c{}".format(self.num_classes))
            self.load_model(model_subdir)
        
        
    def to_train(self, classes_n, epochs_n):
        # self.create_model()
        self.model = self.build(13, 44, 1, classes_n, l2(0.0005))
        
        self.model.summary()
        datadir = 'prep_dataset'
        self.labels = self.get_names(datadir, classes_n)
        print(self.labels)
        validation, testing, training = load_data(datadir, self.labels)
        
        self.train(epochs_n, self.prep_data(training), self.prep_data(validation), self.models_dir)
        
        
    def load_model(self, model_path):
        try:
            self.model = load_model(model_path)
            with open(os.path.join(model_path, 'labels.txt')) as file:
                lines = file.readlines()
                self.labels = [line.rstrip() for line in lines]
        except:
            print("There is no trained model in", model_path)
            self.to_train()

    
    def build(self, height, width, depth, classes, reg, init="he_normal"):
        # initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
		# if we are using "channels first", update the input shape
		# and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        
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
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        model.compile(loss='categorical_crossentropy', 
                            optimizer="adam",
                            metrics=['accuracy'])

        return model

    def get_names(self, datadir, num):
        names = os.listdir(datadir)[:num]
        return names

    def prep_data(self, data):
        x = np.expand_dims(np.concatenate(data['mfcc']), axis=-1)
        y = np_utils.to_categorical(np.concatenate(data['labels']), num_classes=self.num_classes)

        return x,y

        
    def train(self, num_epochs, train_data, valid_data, result_dir):
        callback = TrainingCallback(result_dir, self.labels, valid_data)
        self.model.fit(train_data[0], train_data[1], 
                       validation_data = valid_data,
                       epochs = num_epochs, 
                       callbacks = [callback])
        
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
            print("I don't know what you said. Maybe ", self.labels[y_pred])
        elif y_pred_perc>0.3 and y_pred_perc<0.6: 
            print("I bet that you said ", self.labels[y_pred])
        elif y_pred_perc>0.6 and y_pred_perc<0.8: 
            print("I'm pretty sure you said ", self.labels[y_pred])
        else:
            print("You said ", self.labels[y_pred])


          
if __name__ == '__main__':
    ai = AudioAI(2, 'Results', True)
    print(ai.labels)
    rec = AudioRecord()
    for i in range(10):
        
        rec.record()
        audio_data = rec.get_live_rec_data()
        ai.predict(audio_data)
        time.sleep(3.0)