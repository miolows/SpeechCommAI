import numpy as np
import pyaudio
import tomli

from speechcommai.wrap import timer

class Record(object):
    def __init__(self, record_queue):
        with open("config.toml", mode="rb") as fp:
            self.config = tomli.load(fp)
        self.rate = self.config['audio']['rate']
        self.record_q = record_queue
        self.format = pyaudio.paInt16
        self.chunk = 3024
        self.channels = 1       

    ''' *** Recording *** '''   
    def callback(self, in_data, frame_count, time_info, status):
        data = np.frombuffer(in_data, dtype=np.int16) / 32768.0
        self.record_q.put(data)
        return (in_data, pyaudio.paContinue)

    def start_recording(self):
        self.pyaudio = pyaudio.PyAudio()
        self.stream = self.pyaudio.open(format=self.format,
                                        channels=self.channels,
                                        rate=self.rate,
                                        input=True,
                                        frames_per_buffer=self.chunk,
                                        stream_callback=self.callback)

    def stop_recording(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pyaudio.terminate()

    
if __name__ == '__main__':
    pass