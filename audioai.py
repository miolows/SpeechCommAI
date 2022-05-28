import numpy as np
import os

from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D

from data import DataLoader
from callbacks import TrainingCallback, PredictionCallback


class AudioAI():
    def __init__(self, num_classes, models_dir, to_train=False):
        self.models_dir = models_dir
        self.num_classes = num_classes
        if to_train:
            self.to_train()
            
        else:
            model_subdir = os.path.join(models_dir, "c{}".format(self.num_classes))
            self.load_model(model_subdir)
        
        
    def to_train(self):
        self.create_model()
        validation, testing, training = self.load_data("mfcc_Dataset")
        #hard coded num of epochs!!!
        self.train(10, self.prep_data(training), self.prep_data(validation), self.models_dir)
        
        
    def load_model(self, model_path):
        try:
            self.model = load_model(model_path)
            with open(os.path.join(model_path, 'labels.txt')) as file:
                lines = file.readlines()
                self.labels = [line.rstrip() for line in lines]
                print(self.labels)
        except:
            print("There is no trained model in", model_path)
            self.to_train()
 
    def create_model(self):
        filters_num = 32
        kernel_s = (3,3)
        input_s = (44,44,1)
        hidden_nodes = 300

        first_activ_func = 'relu'
        hidden_activ_func = 'relu'
        last_activ_func = 'softmax'
        
        first_conv_layer = Conv2D(filters = filters_num,
                                  kernel_size = kernel_s,
                                  activation = first_activ_func,
                                  input_shape = input_s)
        hidden_layer = Dense(hidden_nodes, activation = hidden_activ_func)
        last_layer = Dense(self.num_classes, activation = last_activ_func)
        
        self.model = Sequential()
        self.model.add(first_conv_layer)
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Flatten())
        self.model.add(Dropout(0.4))
        self.model.add(hidden_layer)
        self.model.add(Dropout(0.4))
        self.model.add(last_layer)
        
        self.model.compile(loss='categorical_crossentropy', 
                           optimizer="adam",
                           metrics=['accuracy'])
        self.model.summary()
        

    def load_data(self, data_dir):
        dl = DataLoader(self.num_classes, data_dir)
        self.labels = dl.class_labels
        return dl.get_data()    #tuple of validation, testing and training data

        
    def prep_data(self, data):
        x_data =  np.expand_dims(np.stack(data[1]), axis=-1)
        y_data = np_utils.to_categorical(data[0], num_classes=self.num_classes)
        return x_data, y_data

        
    def train(self, num_epochs, train_data, valid_data, result_dir):
        callback = TrainingCallback(result_dir, self.labels, valid_data)
        self.model.fit(train_data[0], train_data[1], 
                       validation_data = valid_data,
                       epochs = num_epochs, 
                       callbacks = [callback])
        
    def predict(self, data):
        callback = PredictionCallback()
        self.model.predict(data, train_data[1], 
                       validation_data = valid_data,
                       epochs = num_epochs, 
                       callbacks = [callback])
        
          
    
if __name__ == '__main__':
    ai = AudioAI(35, 'Results')
    # validation, testing, training = ai.load_data("mfcc_Dataset")
    # #hard coded num of epochs!!!
    # ai.train(10, ai.prep_data(training), ai.prep_data(validation), ai.models_dir)
        
# data_dir = "Dataset"
# prep_data_dir = "mfcc_Dataset"
# result = "Results"
# nr_of_classes = 2

# epo = 10
# hidden_nodes = 300
# first_layer_conv_width = 3
# first_layer_conv_height = 3

# dl = DataLoader(nr_of_classes, prep_data_dir)
# audio_labels = dl.class_labels
# validation, testing, training = dl.get_data()

# y_valid = np_utils.to_categorical(validation[0], num_classes=nr_of_classes)
# y_train = np_utils.to_categorical(training[0], num_classes=nr_of_classes)
# y_test = np_utils.to_categorical(testing[0], num_classes=nr_of_classes)

# x_valid = np.expand_dims(np.stack(validation[1]), axis=-1)
# x_train = np.expand_dims(np.stack(training[1]), axis=-1)
# x_test = np.expand_dims(np.stack(testing[1]), axis=-1)

# # ai = AudioAI(nr_of_classes, epo, hidden_nodes, (first_layer_conv_width, first_layer_conv_height))
# # ai.learn((x_train,y_train), (x_valid,y_valid), result, audio_labels)



# # create model
# model = Sequential()
# model.add(Conv2D(32,
#                   kernel_size=(first_layer_conv_width, first_layer_conv_height),
#                   input_shape=(44,44,1),
#                   activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Flatten())
# model.add(Dropout(0.4))
# model.add(Dense(hidden_nodes, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(nr_of_classes, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer="adam",
#               metrics=['accuracy'])

# model.summary()

# # Fit the modelmodel.summary
# history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
#           epochs=epo, callbacks=[TrainingCallback(result, audio_labels, (x_valid, y_valid))])




# # Evaluate the model on the test data using `evaluate`
# print("Predict on test data")
# results = model.predict(x_test)
# y_pred = np.argmax(results, axis=-1)
# y_true = y_test.argmax(axis=1)
# plot_title = 'predict'
# cm = ConfMatrix(plot_title, audio_labels, y_true, y_pred)
# # cm = ConfMatrix('Prediction', audio_labels, y_true, y_pred)
# cm.draw('CMRmap', 'Wyniki/predict')
