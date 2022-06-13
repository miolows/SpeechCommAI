import os

class DirMaker():
    def __init__(self, labels, prepdir, path=None):
        self.prepdir = prepdir
        self.audio_labels = labels
        self.path = path
        
        if self.path is None:
            self.path = os.path.join(os.getcwd(), self.prepdir)
        self.make_dir(self.path)
        
        for l in self.audio_labels:
            self.make_label_dir(l)
        

    def make_dir(self, dirpath):
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)


    def make_label_dir(self, label):
        label_path = os.path.join(self.path, label)
        self.make_dir(label_path)
        

# curr_dir = os.getcwd()
# audio_labels = os.listdir('Dataset/') #[:classes]
# print(audio_labels)

# created_dir = "prep_dataset"
# path = os.path.join(curr_dir, created_dir)
# if not os.path.exists(path):
#    os.makedirs(path)