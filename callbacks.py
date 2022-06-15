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
        self.learning_frames = []
        self.valid_acc = 0
        
        
    def on_train_begin(self, logs=None):
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)      
                    
        with open(os.path.join(self.result_dir, 'labels.txt'), 'w') as fp:
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
        
        epoch_acc = max(self.valid_acc, logs['val_accuracy'])
        if self.valid_acc == epoch_acc:
            pass
        else:
            self.valid_acc = epoch_acc
            print("Change of valid_acc: ", epoch_acc)
            self.model.save(self.result_dir)
            
        # self.learning_frames.append(cm.fig)
        
        
    def on_train_end(self, logs=None):
        # self.model.save(self.result_dir)
        print(logs)
        
        # # Plot and save training & validation accuracy values
        # HistoryPlot('Wyniki/', history, 'accuracy', 'Model accuracy', 'Accuracy')
        # # Plot and save training & validation loss values
        # HistoryPlot('Wyniki/', history, 'loss', 'Model loss', 'Loss')


class PredictionCallback(Callback):
    
    pass