from keras.callbacks import Callback
from tensorflow.keras.models import save_model
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
        self.max_val_acc = 0
        self.confusion_matrix = ConfMatrix(labels, result_dir)
        
    def on_train_begin(self, logs=None):
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)      
                    
        with open(os.path.join(self.result_dir, 'labels.txt'), 'w') as fp:
            for label in self.labels:
                fp.write("%s\n" % label)
        
       
    def on_epoch_end(self, epoch, logs={}):
        val_predict = self.model.predict(self.validation_data[0])
        y_pred = np.argmax(val_predict, axis=-1)
        y_true = self.validation_data[1].argmax(axis=1)

        self.confusion_matrix.update(y_true, y_pred)
        draw_title = f'Confusion Matrix of Epoch: {epoch}'
        self.confusion_matrix.draw(draw_title)
        
        if self.max_val_acc < logs['val_accuracy']:
            self.max_val_acc = logs['val_accuracy']
            print()
            print(f"Change of acc: {self.max_val_acc:.3f}")
            
            save_title = f"Confusion Matrix (accuracy = {100*self.max_val_acc:.1f}%)"
            self.confusion_matrix.save(save_title)
            save_model(self.model, self.result_dir)

    def on_train_end(self, logs=None):
        print(logs)
        
        # # Plot and save training & validation accuracy values
        # HistoryPlot('Wyniki/', history, 'accuracy', 'Model accuracy', 'Accuracy')
        # # Plot and save training & validation loss values
        # HistoryPlot('Wyniki/', history, 'loss', 'Model loss', 'Loss')


class PredictionCallback(Callback):
    def __init__(self):
        super().__init__()
        
    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop predicting; got log keys: {}".format(keys))
        print(logs['outputs'])

        
        # pass
    
        # x_data =  np.expand_dims(data, axis=0)
        # x_data =  np.expand_dims(x_data, axis=-1)
        # predict = self.model.predict(x_data)
        # y_pred = np.argmax(predict, axis=-1)[0]
        # y_pred_perc = np.squeeze(predict)[y_pred]
        # # print(predict)
        # # print(y_pred_perc)
        # # list(map(lambda x: self.labels))
        # if y_pred_perc<0.3:
        #     print("I don't know what you said. Maybe {}".format(self.class_names[y_pred]))
        # elif y_pred_perc>0.3 and y_pred_perc<0.6: 
        #     print("I bet that you said {}".format(self.class_names[y_pred]))
        # elif y_pred_perc>0.6 and y_pred_perc<0.8: 
        #     print("I'm pretty sure you said {}".format(self.class_names[y_pred]))
        # else:
        #     print("You said {}".format(self.class_names[y_pred]))

    
    pass