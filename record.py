import numpy as np
import pyaudio
import time
import librosa
import librosa.display
import wave
import matplotlib.pyplot as plt


class AudioRecord(object):
    def __init__(self):
        self.FORMAT = pyaudio.paInt16
        self.RATE = 22050
        self.CHUNK = 1024
        self.RECORD_SECONDS = 2
        self.CHANNELS = 2
        self.WAVE_OUTPUT_FILENAME = "output.wav"


    def record(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)
        
        print("* recording")

        frames = []

        for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            data = stream.read(self.CHUNK)
            frames.append(data)

        print("* done recording")
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        self.save(p.get_sample_size(self.FORMAT), frames)


    def save(self, sample_size, frames):
        wf = wave.open(self.WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(sample_size)
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()


    def get_live_rec_data(self):
        y, _ = librosa.load(self.WAVE_OUTPUT_FILENAME, sr=self.RATE)
        
        split = librosa.effects.split(y)
        samples_num = split.shape[0]
        # fig, ax = plt.subplots(nrows=(samples_num+1), sharex=True)
        # librosa.display.waveshow(y, sr=self.RATE, ax=ax[0])


        for s in range(samples_num):
            s_start = split[s,0]
            s_stop = split[s,1]
            s_len = s_stop - s_start
            sample = y[s_start:s_stop]
            
            if s_len<self.RATE:
                middle_b = int(0.5*(self.RATE-s_len))
                middle_e = int(0.5*(self.RATE+s_len))
                norm_sample = np.zeros(self.RATE)
                norm_sample[middle_b:middle_e] = sample
                
                # Compute MFCC features from the raw signal
                mfcc = librosa.feature.mfcc(y=norm_sample, sr=self.RATE, n_mfcc=44)
                
                #Normalization of data (min=0.0; max=1.0)
                mfcc_min = mfcc.min()
                mfcc = np.array(list(map(lambda x: x-mfcc_min, mfcc)))
                mfcc_max = mfcc.max()
                mfcc = np.array(list(map(lambda x: x/mfcc_max, mfcc)))
                
                # librosa.display.waveshow(norm_sample, sr=self.RATE, ax=ax[s+1])

                #returns only 1st execution
                return mfcc
                
                
            else:
                print("too long sample")


if __name__ == '__main__':
    audio = AudioRecord()
    audio.record()
    g = audio.get_live_rec_data()

    

# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# import numpy as np
# import pyaudio
# import time
# import librosa

# class AudioRecord(object):
#     def __init__(self):
#         self.FORMAT = pyaudio.paFloat32
#         self.CHANNELS = 1
#         self.RATE = 22050
#         self.CHUNK = 22050
#         self.p = None
#         self.stream = None
#         self.audio_data = None

#     def start(self):
#         self.p = pyaudio.PyAudio()
#         self.stream = self.p.open(format=self.FORMAT,
#                                   channels=self.CHANNELS,
#                                   rate=self.RATE,
#                                   input=True,
#                                   output=False,
#                                   stream_callback=self.callback,
#                                   frames_per_buffer=self.CHUNK)

#     def stop(self):
#         self.stream.close()
#         self.p.terminate()

#     def callback(self, in_data, frame_count, time_info, flag):
#         y = np.frombuffer(in_data, dtype=np.float32)
#         mfcc = librosa.feature.mfcc(y=y, n_mfcc=44)
        
#         mfcc_min = mfcc.min()
#         mfcc = np.array(list(map(lambda x: x-mfcc_min, mfcc)))
#         mfcc_max = mfcc.max()
#         mfcc = np.array(list(map(lambda x: x/mfcc_max, mfcc)))
#         self.audio_data = mfcc
#         return None, pyaudio.paContinue
    
    
    

#     def mainloop(self):
#         time.sleep(1.0)
#         # while (self.stream.is_active()): # if using button you can set self.stream to 0 (self.stream = 0), otherwise you can use a stop condition
#         #     time.sleep(2.0)


#     def record(self):
#         self.start()
#         print("Talk to me...")
#         time.sleep(1.5)    
#         self.stop()
#         print("Stop")

# if __name__ == '__main__':

#     audio = AudioRecord()
#     audio.start()     # open the the stream
#     audio.mainloop()
#     audio.stop()



# # import pyaudio
# # import struct
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import time

# # %matplotlib tk

# # CHUNK = 1024 * 4
# # FORMAT = pyaudio.paInt16
# # CHANNELS = 1
# # RATE = 44100


# # audio = pyaudio.PyAudio()

# # stream = audio.open(format = FORMAT,
# #                     channels = CHANNELS,
# #                     rate = RATE,
# #                     input = True,
# #                     output = True,
# #                     frames_per_buffer = CHUNK)



# # fig, ax = plt.subplots()
# # x = np.arange(0, 2*CHUNK, 2)
# # line, = ax.plot(x, np.zeros((CHUNK)))
# # ax.set_ylim(-5*255,5*255)
# # ax.set_xlim(0, CHUNK)
# # while True:
# #     data = stream.read(CHUNK)
# #     int_data = struct.unpack( str( CHUNK ) + 'h', data )

# #     line.set_ydata(int_data)
# #     fig.canvas.draw()
# #     fig.canvas.flush_events()
    


# # # import pyaudio
# # # import wave
# # # import sys


# # # audio = pyaudio.PyAudio()

# # # stream = audio.open(format = pyaudio.paInt16,
# # #                     channels = 1,
# # #                     rate = 44100,
# # #                     input = True,
# # #                     frames_per_buffer = 1024)

# # # frames = []

# # try:
# #     while True:
# #         data = stream.read(1024)
# #         frames.append(data)
# #         print(data)
# # except KeyboardInterrupt:
# #     print("Audio acquisition stopped.")
    
    
# # # stream.stop_stream()
# # # stream.close()
# # # audio.terminate()

# # # sound_file = wave.open("record.wav", "wb")
# # # sound_file.setnchannels(1)
# # # sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
# # # sound_file.setframerate(44100)
# # # sound_file.writeframes(b''.join(frames))
# # # sound_file.close()
