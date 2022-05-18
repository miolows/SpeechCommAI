import numpy as np
#import librosa
import matplotlib.pyplot as plt
import os
from keras.callbacks import Callback
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D


from plot import ConfMatrix, HistoryPlot
from data import DataLoader


# ********** CNN ********** #
class PredictionCallback(Callback):
    def __init__(self, labels, val_data):
        super().__init__()
        self.validation_data = val_data
        self.labels = labels  
       
    def on_epoch_end(self, epoch, logs={}):
        predict = model.predict(self.validation_data[0])
        y_pred = np.argmax(predict, axis=-1)
        y_true = self.validation_data[1].argmax(axis=1)
        plot_title = 'Confusion Matrix of Epoch: ' + str(epoch)
        cm = ConfMatrix(plot_title, self.labels, y_true, y_pred)
        cm.draw('CMRmap', 'Wyniki/CMTRX' + str(epoch))
        
        
    def on_train_end(self, logs=None):
        
        print(logs)
        
        # # Plot and save training & validation accuracy values
        # HistoryPlot('Wyniki/', history, 'accuracy', 'Model accuracy', 'Accuracy')
        # # Plot and save training & validation loss values
        # HistoryPlot('Wyniki/', history, 'loss', 'Model loss', 'Loss')


class AudioAI():
    def __init__(self, class_num, epoch, hidden_nodes, conv_size):
        self.class_num = class_num
        self.epoch_num = epoch
        self.hidden_nodes = hidden_nodes
        self.conv_size = conv_size
        
    def create_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, self.conv_size,
                          input_shape=(44,44,1),
                          activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Flatten())
        self.model.add(Dropout(0.4))
        self.model.add(Dense(self.hidden_nodes, activation='relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(self.class_num, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer="adam",
                      metrics=['accuracy'])

        self.model.summary()
        
        
    def learn(self, train_data, valid_data):
        self.model.fit(train_data[0], train_data[1], validation_data=(valid_data[0], valid_data[1]),
                  epochs=epo, callbacks=[PredictionCallback(audio_labels, (valid_data[0], valid_data[1]))])
        
        
data_dir = "Dataset"
prep_data_dir = "mfcc_Dataset"
nr_of_classes = 2

epo = 10
hidden_nodes = 300
first_layer_conv_width = 3
first_layer_conv_height = 3

dl = DataLoader(nr_of_classes, prep_data_dir)
audio_labels = dl.class_labels
validation, testing, training = dl.get_data()

y_valid = np_utils.to_categorical(validation[0], num_classes=nr_of_classes)
y_train = np_utils.to_categorical(training[0], num_classes=nr_of_classes)
y_test = np_utils.to_categorical(testing[0], num_classes=nr_of_classes)

x_valid = np.expand_dims(np.stack(validation[1]), axis=-1)
x_train = np.expand_dims(np.stack(training[1]), axis=-1)
x_test = np.expand_dims(np.stack(testing[1]), axis=-1)

# create model
model = Sequential()
model.add(Conv2D(32,
                 kernel_size=(first_layer_conv_width, first_layer_conv_height),
                 input_shape=(44,44,1),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(hidden_nodes, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(nr_of_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer="adam",
              metrics=['accuracy'])

model.summary()

# Fit the modelmodel.summary
history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
          epochs=epo, callbacks=[PredictionCallback(audio_labels, (x_valid, y_valid))])




# Evaluate the model on the test data using `evaluate`
print("Predict on test data")
results = model.predict(x_test)
y_pred = np.argmax(results, axis=-1)
y_true = y_test.argmax(axis=1)
plot_title = 'predict'
cm = ConfMatrix(plot_title, audio_labels, y_true, y_pred)
# cm = ConfMatrix('Prediction', audio_labels, y_true, y_pred)
cm.draw('CMRmap', 'Wyniki/predict')
