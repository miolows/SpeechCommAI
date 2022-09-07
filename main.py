from speechcommai.config import Configurator
from speechcommai.ai.cnn import AudioAI
from speechcommai.audio.live import live_record

from speechcommai.wrap import timer

if __name__ == '__main__':
    
    config = Configurator()
    data_collection = 'all'
    ai = AudioAI(config, data_collection)
    
    live_record(config, ai)