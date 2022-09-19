import tensorflow_datasets as tfds
import os
import shutil
from speechcommai.wrap import mkdir

def move_dirs(beg_dir, dest_dir):    
    for index, d in enumerate(os.walk(beg_dir)):
        root, dirs, files = d
        if index > 1:
            if not os.path.basename(root) == '_background_noise_':
                shutil.move(root, dest_dir)

@mkdir
def download(dataset_dir, download = 'Download', extract = 'Temp'):
    dl_manager = tfds.download.DownloadManager(download_dir = download, extract_dir = extract)
    dl_manager.download_and_extract('http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz')

    shutil.rmtree(download)
    move_dirs(extract, dataset_dir)
    shutil.rmtree(extract)


