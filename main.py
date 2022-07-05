import prep
from config import Configurator
from record import AudioRecord
from cnn import AudioAI
from multiprocessing import Queue, Process
import time

record_queue = Queue()
mfcc_queue = Queue()

def predict_live_speech():
    audio_data = mfcc_queue.get()
    ai.predict(audio_data)


if __name__ == '__main__':
    config = Configurator()
    rate = config.get('audio', 'rate')
    mfcc_n = config.get('audio', 'mfcc coefficients')
    duration = config.get('audio', 'prep duration')    
    data_collection = 'all'
    threshold = 0.5
    ai = AudioAI(config, data_collection)
    audio = AudioRecord(config, record_queue)
    audio.start_recording()
    
    audio_processing = Process(target=prep.process_live_record, args=(record_queue, mfcc_queue, threshold, rate, duration, mfcc_n))
    audio_processing.daemon = True
    audio_processing.start()
    
    try:
        print('Talk to me...')
        while True:
            if not mfcc_queue.empty():
                predict_live_speech()

    except KeyboardInterrupt:
        print("End")
        audio.stop_recording()

        
    # audio.save_history()