import prep
from config import Configurator
from record import AudioRecord
from cnn import AudioAI
import queue
import time


if __name__ == '__main__':
    config = Configurator()
    data_collection = 'all'
    ai = AudioAI(config, data_collection)
    audio = AudioRecord(config)
    audio.start_recording()
    
    try:
        print('Talk to me...')
        while True: 
            if audio.data_available():
                audio_data = audio.get_audio_data()
                ai.predict(audio_data)
    except KeyboardInterrupt:
        print("End")
        audio.stop_recording()
        # words_queue.join()
        
    audio.save_history()