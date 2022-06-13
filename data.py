import os
import re
import time
import hashlib
import functools
import numpy as np

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer



class DataLoader():
    def __init__(self, class_num, data_dir):
        self.data_directory = data_dir
        self.class_number = class_num
        self.class_labels = self.getLabels(class_num, data_dir)
        self.max_per_class = 2**27 - 1  # ~134M
        print(self.class_labels)
        
        self.validation = []
        self.testing = []
        self.training = []
        
        self.valid_perc = 20
        self.test_perc = 20
    
        self.init_data()
    
    
    def init_data(self):
        split_data = {'validation':   self.validation,
                      'training':     self.training,
                      'testing':      self.testing}
        
        for class_idx, class_dir in enumerate(self.class_labels):
            print(class_idx)
            dir_path = os.path.join(self.data_directory, class_dir)
            audio_files = os.listdir(dir_path)
            for file_iter, audio_file in enumerate(audio_files):
                which = self.which_set(audio_file, self.valid_perc, self.test_perc)
                data_file = os.path.join(dir_path, audio_file)
                data = np.loadtxt(data_file)
                # print(type(data))
                # data = pd.read_csv(data_file, sep=" ", header = None)

                labeled_data = [class_idx, data]
                split_data[which].append(labeled_data)
               
        self.validation = np.array(self.validation, dtype=object)
        self.testing = np.array(self.testing, dtype=object)
        self.training = np.array(self.training, dtype=object)
    
    
    def getLabels(self, num, data_dir):
        dir_path = os.path.join(os.getcwd(), data_dir)
        labels = os.listdir(dir_path)[:num]
        return labels
    
    
    def split_labels(self, arr):
        if len(arr):
            return arr[:,0], arr[:,1] #return labels, data
        else:
            return [], []
    

    def get_data(self):
        val_data = self.split_labels(self.validation)
        test_data = self.split_labels(self.testing)
        train_data = self.split_labels(self.training)
        return val_data, test_data, train_data

    
    def which_set(self, filename, validation_percentage, testing_percentage):

        """Determines which data partition the file should belong to.
        This is a slightly changed algorithm found in licence file of used dataset
        """
        base_name = os.path.basename(filename)
        # We want to ignore anything after '_nohash_' in the file name when
        # deciding which set to put a wav in, so the data set creator has a way of
        # grouping wavs that are close variations of each other.
        hash_name = re.sub(r'_nohash_.*$', '', base_name)
        
        # This looks a bit magical, but we need to decide whether this file should
        # go into the training, testing, or validation sets, and we want to keep
        # existing files in the same set even if more files are subsequently
        # added.
        # To do that, we need a stable way of deciding based on just the file name
        # itself, so we do a hash of that and then use that to generate a
        # probability value that we use to assign it.
        h = hashlib.sha1()
        h.update(bytes(hash_name, encoding='utf-8'))
        hash_name_hashed = h.hexdigest()
        percentage_hash = ((int(hash_name_hashed, 16) %
                            (self.max_per_class + 1)) *
                            (100.0 / self.max_per_class))
        if percentage_hash < validation_percentage:
          result = 'validation'
        elif percentage_hash < (testing_percentage + validation_percentage):
          result = 'testing'
        else:
          result = 'training'
        return result
        
    
if __name__ == '__main__':
    
    dl = DataLoader(3, 'mfcc_Dataset')
    a, b, c = dl.get_data()
    # a, b, c = dl.tralala()

    

    
    