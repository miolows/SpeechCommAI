from keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np
import os

from plot import ConfMatrix, HistoryPlot


class TrainingCallback(Callback):
    def __init__(self, result_dir, labels, val_data):
        super().__init__()
        self.result_dir = result_dir
        self.validation_data = val_data
        self.labels = labels 
        self.train_dir = os.path.join(self.result_dir, "c{}".format(len(self.labels)))
        self.learning_frames = []
        
    def on_train_begin(self, logs=None):
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)      
        
        if not os.path.exists(self.train_dir):
            os.mkdir(self.train_dir)
            
        with open(os.path.join(self.train_dir, 'labels.txt'), 'w') as fp:
            for label in self.labels:
                fp.write("%s\n" % label)
        
       
    def on_epoch_end(self, epoch, logs={}):
        predict = self.model.predict(self.validation_data[0])
        y_pred = np.argmax(predict, axis=-1)
        y_true = self.validation_data[1].argmax(axis=1)
        print(len(y_pred), len(y_true))
        plot_title = 'Confusion Matrix of Epoch: ' + str(epoch)
        cm = ConfMatrix(plot_title, self.labels, y_true, y_pred)
        cm.draw('CMRmap', 'aaa')
        # self.learning_frames.append(cm.fig)
        
        
    def on_train_end(self, logs=None):
        self.model.save(self.train_dir)
        print(logs)
        
        # # Plot and save training & validation accuracy values
        # HistoryPlot('Wyniki/', history, 'accuracy', 'Model accuracy', 'Accuracy')
        # # Plot and save training & validation loss values
        # HistoryPlot('Wyniki/', history, 'loss', 'Model loss', 'Loss')


class PredictionCallback(Callback):
    
    pass