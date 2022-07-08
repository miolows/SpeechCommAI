import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties
from sklearn.metrics import confusion_matrix

class HistoryPlot():
    def __init__(self, output, data, param, title, ylabel, xlabel='Epoch'):
        fig = plt.figure()
        plt.plot(data.history[param])
        plt.plot(data.history['val_'+param])
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.xticks(range(len(data.history[param])))
        plt.legend(['Train', 'Test'], loc='upper left')
        fig.savefig(output + title + '.png', dpi=199) 
        fig.autofmt_xdate()
        plt.show()


class ConfMatrix():
    def __init__(self, words_label, save_path, cmap='CMRmap'):
        self.labels = words_label
        self.m_len = len(words_label)
        self.color_map = cmap
        self.save_path = save_path
        
        self.y_data = np.zeros(self.m_len)
        self.x_data = np.zeros(self.m_len)

        self.title_fontsize = 17
        self.title_pad = 20
        self.labels_fontsize = 15
        self.labels_pad = 20
        self.ticks_fontsize = self.ticks_fsize()
        
        self.px_val_decimal = self.pixel_vdecimal()
        self.px_val_size = self.pixel_vsize()
        
        
    def update(self, y, x):
        self.y_data = y
        self.x_data = x


    def save(self, title):
        self.draw(title, to_save=True)
        
        file_name = 'Confusion Matrix.png'
        path = os.path.join(self.save_path, file_name)
        plt.savefig(path, dpi=150, bbox_inches='tight')


    def draw(self, title, to_save=False):
        plt.figure()
        matrix = confusion_matrix(self.y_data, self.x_data, normalize='pred')
        
        plt.imshow(matrix, cmap=self.color_map, norm=Normalize(0,1))
        self.write_on_pixels(matrix)
        plt.title(title, 
                  fontsize=self.title_fontsize, pad=self.title_pad)
        plt.ylabel('True class',
                   fontsize=self.labels_fontsize)
        plt.xlabel('Predicted class',
                   fontsize=self.labels_fontsize)
        plt.yticks(np.arange(self.m_len), self.labels, 
                   fontsize=self.ticks_fontsize)        
        plt.xticks(np.arange(self.m_len), self.labels, 
                   fontsize=self.ticks_fontsize, rotation='vertical')
        
        plt.colorbar()
        if not to_save:
            plt.show()
            
        # fig.clear()
        
        
    def linear(self, x, y_0, x_0, v=(0,0)):
        ''' 
        y_0 - y(x=0)
        x_0 - y(m)=0
        v - shift vector
        '''
        b = y_0         
        a = -(y_0/x_0)
        y = (a*(x-v[0]) + b) + v[1]
        return y
    
    
    def ticks_fsize(self):
        start_s = 15
        l_max = 60
        
        s = self.linear(self.m_len, start_s, l_max)
        size = max(s, 0)
        return size

        
    def pixel_vdecimal(self):
        start_dec = 2
        l_max = 20
        
        d = int(np.round(self.linear(self.m_len, start_dec, l_max)))
        dec = max(d, 1)
        decimal = f'%.{dec}f'
        return decimal
    
    
    def pixel_vsize(self):
        start_s = 20
        l_max = 20
        s = self.linear(self.m_len, start_s, l_max)
        size = max(s, 0)
        return size 
    
    
    def write_on_pixels(self, matrix, txt_col=['white', 'black']):
        if self.px_val_size: #if the value size is not 0
            for index, val in np.ndenumerate(matrix):
                y, x = index
                txt = self.px_val_decimal % val #'%.Xf' % val (% is not a modulo)
                col = txt_col[int(np.round(val))]
                plt.text(x, y, txt, ha="center", va="center", 
                         fontproperties=FontProperties(size=self.px_val_size), color=col)