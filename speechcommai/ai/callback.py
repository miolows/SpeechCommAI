from keras.callbacks import Callback
from tensorflow.keras import models
import numpy as np
import os

from speechcommai.data.plot import ConfMatrix, plot_history


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

        
    def on_epoch_end(self, epoch, logs={}):
        val_predict = self.model.predict(self.validation_data[0])
        y_pred = np.argmax(val_predict, axis=-1)
        y_true = self.validation_data[1].argmax(axis=1)

        self.confusion_matrix.update(y_true, y_pred)
        draw_title = f'Confusion Matrix of Epoch: {epoch+1}'
        self.confusion_matrix.draw(draw_title)
        
        if self.max_val_acc < logs['val_accuracy']:
            self.max_val_acc = logs['val_accuracy']
            print()
            print(f"Change of acc: {self.max_val_acc:.3f}")
            
            save_title = f"Confusion Matrix (accuracy = {100*self.max_val_acc:.1f}%)"
            self.confusion_matrix.save(save_title)
            models.save_model(self.model, self.result_dir)

    def on_train_end(self, logs=None):
        # Plot and save training & validation accuracy and loss values
        plot_history(self.result_dir, self.model.history)