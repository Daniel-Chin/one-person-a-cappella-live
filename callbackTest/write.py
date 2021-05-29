print('importing...')
import pyaudio
from time import sleep
from threading import Lock

print('Preparing...')
PAGE_LEN = 256

SR = 22050
DTYPE = pyaudio.paInt32

streamOutContainer = []
terminate_flag = 0
terminateLock = Lock()

def main():
    global terminate_flag
    terminateLock.acquire()
    pa = pyaudio.PyAudio()
    streamOutContainer.append(pa.open(
        format = DTYPE, channels = 1, rate = SR, 
        output = True, frames_per_buffer = PAGE_LEN,
    ))
    streamIn = pa.open(
        format = DTYPE, channels = 1, rate = SR, 
        input = True, frames_per_buffer = PAGE_LEN,
        stream_callback = onAudioIn, 
    )
    streamIn.start_stream()
    try:
        while streamIn.is_active():
            sleep(1)
    except KeyboardInterrupt:
        print('Ctrl+C received. Shutting down. ')
    finally:
        print('Releasing resources... ')
        terminate_flag = 1
        terminateLock.acquire()
        terminateLock.release()
        streamOutContainer[0].stop_stream()
        streamOutContainer[0].close()
        while streamIn.is_active():
            sleep(.1)   # not perfect
        streamIn.stop_stream()
        streamIn.close()
        pa.terminate()
        print('Resources released. ')

def onAudioIn(in_data, sample_count, *_):
    global terminate_flag

    try:
        if terminate_flag == 1:
            terminate_flag = 2
            terminateLock.release()
            print('PA handler terminating. ')
            # Sadly, there is no way to notify main thread after returning. 
            return (None, pyaudio.paComplete)

        if sample_count > PAGE_LEN:
            print('Discarding audio page!')
            in_data = in_data[-PAGE_LEN:]

        streamOutContainer[0].write(in_data, PAGE_LEN)
        return (None, pyaudio.paContinue)
    except:
        terminateLock.release()
        import traceback
        traceback.print_exc()
        return (None, pyaudio.paAbort)

main()
