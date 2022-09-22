import tomli
import os, sys
from speechcommai.ai.cnn import CNN
from speechcommai.audio.live import live_record
from speechcommai.data.preprocessing import prep_dataset
from speechcommai.data.load import load_data
from speechcommai.data.download import download

from speechcommai.wrap import timer


def get_from_config(*args):
    with open("config.toml", mode="rb") as fp:
        config = tomli.load(fp)
    return [config[v] for v in args]

def download_speech_data():
    dirs = get_from_config('directories')[0]
    dataset = dirs['dataset']
    download(dataset)

def preprocessing():
    dirs, audio = get_from_config('directories', 'audio')
    
    raw = dirs['dataset']
    prep = dirs['preprocessed_data']
    
    sample_rate = audio['rate']
    duration = audio['duration']
    n_mfcc = audio['mfcc']
    hop_l = audio['hop_length']
    
    prep_dataset(prep, raw, 
                 sample_rate, duration, n_mfcc, hop_l,
                 training=80, validation=20)

def init_ai(collection='all'):
    dirs, audio, data = get_from_config('directories', 'audio', 'data')
    class_names = data[collection]
    model_dir = dirs['saved_models']
    model_path = os.path.join(model_dir, collection)
    
    shape0 = audio['mfcc']
    shape1 = int(audio['duration'] * audio['rate'] / audio['hop_length']) + 1
    input_shape=(shape0, shape1, 1)
    
    return CNN(model_path, class_names, input_shape)

def train_ai(collection='all'):
    ai = init_ai(collection)
    dirs, data = get_from_config('directories', 'data')
    prep_data_dir = dirs['preprocessed_data']
    class_names = data[collection]
    
    train_data = load_data(prep_data_dir, class_names, 'training')
    valid_data = load_data(prep_data_dir, class_names, 'validation')

    ai.train_model(train_data, valid_data)

def live(collection='all'):
    ai = init_ai(collection)
    ai.load_model()
    live_record(ai)
    
def print_menu():
    print("""
          Menu
          D - Download the raw dataset
          P - Pre-process the dataset
          T - Train the model
          L - Live record
          """)     
    
    
    
if __name__ == '__main__':
    menu = {'D': download_speech_data,
            'P': preprocessing,
            'T': train_ai,
            'L': live,
            'default': print_menu}
    
    try:
        command = sys.argv[1]
    except IndexError:
        command = ""
        
    menu.get(command, menu.get('default'))(*sys.argv[2:])