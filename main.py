import prep
from config import Configurator
from record import AudioRecord
from cnn import AudioAI
import queue
import time

# def predict():
#     audio_data = words_queue.get()
#     ai.predict(audio_data)

if __name__ == '__main__':
    words_queue = queue.Queue()

    config = Configurator()
    data_collection = 'all'
    ai = AudioAI(config, data_collection)
    audio = AudioRecord(config, words_queue)
    audio.start_recording()
    
    try:
        print('Talk to me...')
        while True:
            audio_data = words_queue.get()
            ai.predict(audio_data)
    except KeyboardInterrupt:
        print("End")
        audio.stop_recording()
        # words_queue.join()
        