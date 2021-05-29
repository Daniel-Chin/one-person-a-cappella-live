print('importing...')
import pyaudio
from time import sleep
from threading import Lock
import numpy as np
from queue import Queue, Empty

print('Preparing...')
PAGE_LEN = 256

SR = 22050
DTYPE = pyaudio.paFloat32
SILENCE = np.zeros((PAGE_LEN, ))

terminate_flag = 0
inLock = Lock()
outLock = Lock()

q = Queue()

def main():
    global terminate_flag
    inLock.acquire()
    outLock.acquire()
    pa = pyaudio.PyAudio()
    streamOut = pa.open(
        format = DTYPE, channels = 1, rate = SR, 
        output = True, frames_per_buffer = PAGE_LEN,
        stream_callback = requestAudioOut, 
    )
    streamIn = pa.open(
        format = DTYPE, channels = 1, rate = SR, 
        input = True, frames_per_buffer = PAGE_LEN,
        stream_callback = onAudioIn, 
    )
    streamIn.start_stream()
    streamOut.start_stream()
    print('go...')
    try:
        while streamIn.is_active() and streamOut.is_active():
            sleep(1)
    except KeyboardInterrupt:
        print('Ctrl+C received. Shutting down. ')
    finally:
        print('Releasing resources... ')
        terminate_flag = 1
        inLock.acquire()
        outLock.acquire()
        while streamIn.is_active() or streamOut.is_active():
            sleep(.1)   # not perfect
        streamIn.stop_stream()
        streamIn.close()
        streamOut.stop_stream()
        streamOut.close()
        pa.terminate()
        print('Resources released. ')

def onAudioIn(in_data, sample_count, *_):
    global terminate_flag

    try:
        if terminate_flag == 1:
            inLock.release()
            print('onAudioIn terminating. ')
            # Sadly, there is no way to notify main thread after returning. 
            return (None, pyaudio.paComplete)

        if sample_count > PAGE_LEN:
            print('Discarding audio page!')
            in_data = in_data[-PAGE_LEN:]

        q.put(in_data)
        return (None, pyaudio.paContinue)
    except:
        inLock.release()
        import traceback
        traceback.print_exc()
        return (None, pyaudio.paAbort)

def requestAudioOut(_, sample_count, *__):
    global terminate_flag

    try:
        if terminate_flag == 1:
            outLock.release()
            print('requestAudioOut terminating. ')
            # Sadly, there is no way to notify main thread after returning. 
            return (None, pyaudio.paComplete)

        if sample_count != PAGE_LEN:
            print('Error f3p98wunfreiu5', sample_count)
            raise ValueError()

        try:
            return (q.get(timeout=1), pyaudio.paContinue)
        except Empty:
            outLock.release()
            print('requestAudioOut terminating. ')
            return (None, pyaudio.paAbort)
    except:
        outLock.release()
        import traceback
        traceback.print_exc()
        return (None, pyaudio.paAbort)

main()
