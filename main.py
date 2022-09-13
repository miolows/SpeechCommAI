import tomli
from speechcommai.ai.cnn import AudioAI
from speechcommai.audio.live import live_record
from speechcommai.data.preprocessing import prep_dataset
from speechcommai.data.load import load_data

from speechcommai.wrap import timer

def preprocessing(**kwargs):
    with open("config.toml", mode="rb") as fp:
        config = tomli.load(fp)
    
    dir_d = config['directories']
    raw = dir_d['dataset']
    prep = dir_d['preprocessed_data']
    
    audio_d = config['audio']
    sample_rate = audio_d['rate']
    duration = audio_d['duration']
    n_mfcc = audio_d['mfcc']
    hop_l = audio_d['hop_length']
    
    prep_dataset(prep, raw, 
                 sample_rate, duration, n_mfcc, hop_l,
                 **kwargs)

def train_ai(collection='all'):
    ai = AudioAI(collection, train=True)


def live_rec(collection='all'):
    ai = AudioAI(collection)
    live_record(ai)
    
    
if __name__ == '__main__':
    preprocessing(training=80, validation=20)
    train_ai()
    # x = load_data('Prep_Dataset', ['bed', 'bird'], 'validation')
