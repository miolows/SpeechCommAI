import numpy as np
#import librosa
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import Normalize
import os
from keras.callbacks import Callback
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize

from plot import ConfMatrix, HistoryPlot
from data import DataLoader

def getLabels(num, data_dir):
    dir_path = os.path.join(os.getcwd(), data_dir)
    labels = os.listdir(dir_path)[:num]
    return labels


def getData(labels, data_dir='mfcc_Dataset'):
    #get the names of mlp classes as the names of directories of corresponding audio files
    
    #nr_of_samples = 60 #do testów
    #np.random.seed(10) #do testów
    # n=1
    # # labels_of_speech_classes = os.listdir(data_dir + '/')[(n-1)*nr_of_classes:n*nr_of_classes]
    # LABELS = labels_of_speech_classes
    # print(labels_of_speech_classes)
    audio_data = []
    
    for class_iter, speech_class in enumerate(labels):
        print(class_iter)
        data_path = data_dir + '/' + speech_class + '/'
        audio_names = os.listdir(data_path) #[:nr_of_samples] #do testów
        
        for file_iter, audio_file in enumerate(audio_names):
            data_file = data_path + '/' + audio_file
            data = np.loadtxt(data_file)
            data = np.append(class_iter, data)
            audio_data.append(data)
    
    audio_data = np.array(audio_data)
    np.random.shuffle(audio_data)
    
    split_samples = int(0.85*audio_data.shape[0])
    train_data = audio_data[:split_samples,1:].reshape(-1,44,44)
    train_labels = audio_data[:split_samples,0]
    test_data = audio_data[split_samples:,1:].reshape(-1,44,44)
    test_labels = audio_data[split_samples:,0]
    print(np.shape(train_data))
    print(np.shape(test_data))
    return (train_data, train_labels, test_data, test_labels)


# ********** CNN ********** #
class PredictionCallback(Callback):
    def __init__(self, labels, val_data, batch_size = 20):
       super().__init__()
       self.validation_data = val_data
       self.batch_size = batch_size
       self.labels = labels
       
       
    def on_epoch_end(self, epoch, logs={}):
        y_pred = np.argmax(model.predict(self.validation_data[0]), axis=-1)
        y_true = self.validation_data[1].argmax(axis=1)
        
        cm = ConfMatrix(epoch, self.labels, y_true, y_pred)
        cm.draw('CMRmap', 'Wyniki/CMTRX' + str(epoch))
       
       
    # def on_epoch_end(self, epoch, logs={}):
    #     y_pred = np.argmax(model.predict(self.validation_data[0]), axis=-1)
    #     y_true = self.validation_data[1].argmax(axis=1)
    #     labels = os.listdir('Dataset/')[:5]
    #     c_matrix = confusion_matrix(y_true, y_pred, normalize='pred')
    #     ConfMatrixImageMaker('Wyniki/', c_matrix, epoch+1)


# def ConfMatrixImageMaker(output, matrix, epoch):
#     data_dir = 'Dataset/'
#     labels = os.listdir(data_dir)[:5]
#     fig = plt.figure()
#     plt.imshow(matrix, cmap='CMRmap', norm=Normalize(0,1))


#     for index, val in np.ndenumerate(matrix):
#         x, y = index
#         txt = "%.2f" % val
#         col = abs(np.round(val) - 1)
#         plt.text(x, y, txt, ha="center", va="center",\
#                  fontproperties=FontProperties(size="xx-small"), color=(col,col,col))
        
#     '''
#     for i, mi in enumerate(matrix):
#         for j, mj in enumerate(mi):
#             if mj>0.5:
#                 plt.text(j, i, ("%.2f" % matrix[i, j]),ha="center", va="center",\
#                          fontproperties=FontProperties(size="xx-small"), color="black")
#             else:
#                 plt.text(j, i, ("%.2f" % matrix[i, j]),ha="center", va="center", \
#                          fontproperties=FontProperties(size="xx-small"), color="white")
              
#     '''
#     plt.title('Confusion Matrix of Epoch: ' + str(epoch))
#     plt.ylabel('True class')
#     plt.yticks(range(0, matrix.shape[0]), labels)
#     plt.xlabel('Predicted class')
#     plt.xticks(range(0, matrix.shape[1]), labels, rotation='vertical')
#     plt.colorbar()
#     fig.savefig(output + 'CMTRX' + str(epoch) + '.png', dpi=199) 
#     plt.show()
    
    





data_dir = "Dataset"
prep_data_dir = "mfcc_Dataset"
nr_of_classes = 2
audio_labels = getLabels(nr_of_classes, data_dir)

epo = 10
hidden_nodes = 300
first_layer_conv_width = 3
first_layer_conv_height = 3


''' ====================================================================== '''

dl = DataLoader(nr_of_classes, prep_data_dir)
validation, testing, training = dl.get_data()

# one hot encode outputs
# y_train = np.asarray(training[0]).astype('int')
# y_test = np.asarray(testing[0]).astype('int')

y_train = np_utils.to_categorical(training[0], num_classes=nr_of_classes)
y_test = np_utils.to_categorical(testing[0], num_classes=nr_of_classes)

x_train = np.expand_dims(np.array(list(map(lambda x: tf.convert_to_tensor(x), training[1]))), axis=-1)
x_test = np.expand_dims(np.array(list(map(lambda x: tf.convert_to_tensor(x), testing[1]))), axis=-1)

# x_train = np.array(list(map(lambda x: tf.convert_to_tensor(x), training[1]))).reshape(len(training[0]), 44, 44, 1)
# x_test = np.array(list(map(lambda x: tf.convert_to_tensor(x), testing[1]))).reshape(len(testing[0]), 44, 44, 1)
#reshape data
# X_train = training[1].reshape(training[1].shape[0], 44, 44, 1)
# X_test = testing[1].reshape(testing[1].shape[0], 44, 44, 1)

# create model
model = Sequential()
model.add(Conv2D(32,
                  (first_layer_conv_width, first_layer_conv_height),
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
history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
          epochs=epo, callbacks=[PredictionCallback(audio_labels, (x_test, y_test))])


# Plot and save training & validation accuracy values
HistoryPlot('Wyniki/', history, 'accuracy', 'Model accuracy', 'Accuracy')
# Plot and save training & validation loss values
HistoryPlot('Wyniki/', history, 'loss', 'Model loss', 'Loss')

''' ====================================================================== '''


# X_train, y_train, X_test, y_test = getData(audio_labels)

# _, img_width, img_height = np.shape(X_train)

# #reshape data
# X_train = X_train.reshape(X_train.shape[0], img_width, img_height, 1)
# X_test = X_test.reshape(X_test.shape[0], img_width, img_height, 1)

# # one hot encode outputs
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# num_classes = y_train.shape[1]

# # create model
# model = Sequential()
# model.add(Conv2D(32,
#                   (first_layer_conv_width, first_layer_conv_height),
#                   input_shape=(44,44,1),
#                   activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Flatten())
# model.add(Dropout(0.4))
# model.add(Dense(hidden_nodes, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(num_classes, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer="adam",
#               metrics=['accuracy'])

# model.summary()

# # Fit the modelmodel.summary
# history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
#           epochs=epo, callbacks=[PredictionCallback(audio_labels, (X_test, y_test))])


# # Plot and save training & validation accuracy values
# HistoryPlot('Wyniki/', history, 'accuracy', 'Model accuracy', 'Accuracy')
# # Plot and save training & validation loss values
# HistoryPlot('Wyniki/', history, 'loss', 'Model loss', 'Loss')


