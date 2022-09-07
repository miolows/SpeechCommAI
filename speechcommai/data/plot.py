import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties
from sklearn.metrics import confusion_matrix
from matplotlib.ticker import MaxNLocator

def plot_history(output, data):
    params=['accuracy', 'loss']
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    
    title = 'Model History'
    fig.subplots_adjust(hspace=0.4, top=0.85)
    fig.suptitle(title, fontsize=15)
    path = os.path.join(output, 'Model history.png')                

    for p, param in enumerate(params):
        y_training = data.history[param]
        x_training = range(1, len(y_training) + 1)
        y_validation = data.history[f'val_{param}']
        x_validation = range(1, len(y_validation) + 1)
        
        axs[p].plot(x_training, y_training, label='training')
        axs[p].plot(x_validation, y_validation, label='validation')

        axs[p].set_ylabel(param)
        axs[p].set_xlabel('Epoch')
        axs[p].xaxis.set_major_locator(MaxNLocator(integer=True))            
        axs[p].grid(True)

    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
              fancybox=True, shadow=True, ncol=2)
    
    plt.savefig(path, dpi=150, bbox_inches='tight')
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
        
        #I tried to prevent showing saved figure, but it doesn't work. 
        #It ends up popping up in the next call of this function, which has to_save=False.
        if to_save:
            plt.ioff()
        else:
            plt.show()
            

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