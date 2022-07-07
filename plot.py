import numpy as np
import matplotlib.cm as cm
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
    def __init__(self, title, words_label, y, x):
        self.title = title
        self.labels = words_label
        self.y_data = y
        self.x_data = x
        
        self.matrix = confusion_matrix(y, x, normalize='pred')
        self.m_len = len(words_label)

        self.fig = plt.figure()

    def save(self, figure, file_name, dpi=199):
        figure.savefig(file_name, dpi=dpi) 

    def draw(self, color_map, fn):
        # cmp = cm.get_cmap(color_map)
        y_label = 'True class'
        x_label = 'Predicted class'
        
        plt.imshow(self.matrix, cmap=color_map, norm=Normalize(0,1))
        
        self.write_on_pixels()

        # ''' Adjust the text size to prevent overlapping. Is there a more organic way to do this? '''
        # if self.m_len < 20:
        #     dec = "%.2f"
        #     if self.m_len >= 10:
        #         dec = "%.1f"
        #     size = max(20 - self.m_len, 1)
        #     self.annotate_matrix(dec, size)
        
        plt.title(self.title)
        plt.ylabel(y_label)
        plt.xlabel(x_label)    
        
        plt.yticks(np.arange(self.matrix.shape[0]), self.labels)
        plt.xticks(np.arange(self.matrix.shape[1]), self.labels, rotation='vertical')
        plt.colorbar()
        self.save(self.fig, fn)
        plt.show()
        
    def linear(self, x_0, y_0, x):
        b = x_0
        a = -(x_0/y_0)
        y = a*x + b
        return y
        
    def pixel_vdecimal(self):
        dec_0 = 2
        l_max = 20
        
        d = int(np.round(self.linear(dec_0, l_max, self.m_len)))
        decimal = max(d, 1)
        dec = f'%.{decimal}f'
        return dec
    
    def pixel_vsize(self):
        size_0 = 20
        l_max = 20
        size = max(self.linear(size_0, l_max, self.m_len), 5)
        return size 
    
    def write_on_pixels(self, txt_col=['white', 'black']):
        val_dec = self.pixel_vdecimal()
        val_size = self.pixel_vsize()
        if val_size: #if the size is not 0
            for index, val in np.ndenumerate(self.matrix):
                y, x = index
                txt = val_dec % val #'%.Xf' % val (% is not a modulo)
                col = txt_col[int(np.round(val))]
                plt.text(x, y, txt, ha="center", va="center", 
                         fontproperties=FontProperties(size=val_size), color=col)

    