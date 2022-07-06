from config import Configurator
from cnn import AudioAI
from live import live_record

from wrap import timer

if __name__ == '__main__':
    
    config = Configurator()
    data_collection = 'all'
    ai = AudioAI(config, data_collection)
    
    live_record(config, ai)
    
   