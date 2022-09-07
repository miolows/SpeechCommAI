import tensorflow_datasets as tfds



def download(dataset_dir='Test/'):
    dl_manager = tfds.download.DownloadManager(download_dir = dataset_dir, extract_dir = dataset_dir)
    dl_manager.download_and_extract('http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz')


if __name__ == '__main__':
    download()