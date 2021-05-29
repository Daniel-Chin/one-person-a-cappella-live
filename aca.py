print('importing...')
import pyaudio
from time import time, sleep
import numpy as np
import scipy
from scipy.interpolate import interp1d
from threading import Lock
import wave
try:
    from interactive import listen
    from yin import yin
    from streamProfiler import StreamProfiler
    from harmonicSynth import HarmonicSynth, Harmonic
except ImportError as e:
    module_name = str(e).split('No module named ', 1)[1].strip().strip('"\'')
    print(f'Missing module {module_name}. Please download at')
    print(f'https://github.com/Daniel-Chin/Python_Lib/blob/master/{module_name}.py')
    input('Press Enter to quit...')
    raise e

print('Preparing...')
PAGE_LEN = 512
N_HARMONICS = 30
MAX_NOTES = 2
DO_SWIPE = True
DO_PROFILE = True
# WRITE_FILE = None
WRITE_FILE = f'demo_{time()}.wav'

MASTER_VOLUME = .2
SR = 22050
NYQUIST = SR // 2
DTYPE_BUF = (np.float32, pyaudio.paFloat32)
DTYPE_IO = (np.int32, pyaudio.paInt32)
TWO_PI = np.pi * 2
HANN = scipy.signal.get_window('hann', PAGE_LEN, True)
SILENCE = np.zeros((PAGE_LEN, ))
PAGE_TIME = 1 / SR * PAGE_LEN
IMAGINARY_LADDER = np.linspace(0, TWO_PI * 1j, PAGE_LEN)
LADDER = np.arange(1, N_HARMONICS + 1)
FMAX = 1600
INT32RANGE = 2**31
INV_INT32RANGE = 1 / 2**31

streamOutContainer = []
terminate_flag = 0
terminateLock = Lock()
profiler = StreamProfiler(PAGE_LEN / SR, DO_PROFILE)
hSynth = None
harmonics = [Harmonic(220, 0)] * (N_HARMONICS * MAX_NOTES)

if DO_PROFILE:
    _print = print
    def print(*a, **k):
        _print()
        _print(*a, **k)

def sft(signal, freq_bin):
    # Slow Fourier Transform
    return np.abs(np.sum(signal * np.exp(IMAGINARY_LADDER * freq_bin))) / PAGE_LEN

def main():
    global terminate_flag, f, hSynth
    terminateLock.acquire()
    hSynth = HarmonicSynth(
        N_HARMONICS * MAX_NOTES, SR, PAGE_LEN, DTYPE_BUF[0], 
        STUPID_MATCH = True, DO_SWIPE = DO_SWIPE, 
        CROSSFADE_RATIO = .3, 
    )
    pa = pyaudio.PyAudio()
    streamOutContainer.append(pa.open(
        format = DTYPE_IO[1], channels = 1, rate = SR, 
        output = True, frames_per_buffer = PAGE_LEN,
    ))
    if WRITE_FILE is not None:
        f = wave.open(WRITE_FILE, 'wb')
        f.setnchannels(1)
        f.setsampwidth(4)   # 32 / 8 = 4
        f.setframerate(SR)
    streamIn = pa.open(
        format = DTYPE_IO[1], channels = 1, rate = SR, 
        input = True, frames_per_buffer = PAGE_LEN,
        stream_callback = onAudioIn, 
    )
    streamIn.start_stream()
    print('Press ESC to quit. ')
    try:
        while streamIn.is_active():
            op = listen(b'\x1b', priorize_esc_or_arrow=True)
            if op == b'\x1b':
                print('Esc received. Shutting down. ')
                break
    except KeyboardInterrupt:
        print('Ctrl+C received. Shutting down. ')
    finally:
        print('Releasing resources... ')
        terminate_flag = 1
        terminateLock.acquire()
        terminateLock.release()
        streamOutContainer[0].stop_stream()
        streamOutContainer[0].close()
        if WRITE_FILE is not None:
            f.close()
        while streamIn.is_active():
            sleep(.1)   # not perfect
        streamIn.stop_stream()
        streamIn.close()
        pa.terminate()
        print('Resources released. ')

def getEnvelope(signal, len_signal):
    f0 = yin(signal, SR, len_signal, fmax=FMAX)
    harmonics_f = np.arange(0, NYQUIST, f0)
    harmonics_a = [sft(signal * HANN, f_bin) for f_bin in harmonics_f / SR * len_signal]
    harmonics_a[0] = harmonics_a[1]
    f = interp1d(harmonics_f, harmonics_a)
    def envelope(x):
        try:
            return f(x)
        except ValueError:
            return 0
    return envelope

def onAudioIn(in_data, sample_count, *_):
    try:
        if terminate_flag == 1:
            terminateLock.release()
            print('PA handler terminating. ')
            # Sadly, there is no way to notify main thread after returning. 
            return (None, pyaudio.paComplete)

        if sample_count > PAGE_LEN:
            print('Discarding audio page!')
            in_data = in_data[-PAGE_LEN:]

        profiler.gonna('in')
        page = np.frombuffer(
            in_data, dtype = DTYPE_IO[0]
        )
        page = np.multiply(page, INV_INT32RANGE, dtype=DTYPE_BUF[0])

        profiler.gonna('getE')
        envelope = getEnvelope(page, PAGE_LEN)

        profiler.gonna('MIDI')
        ...
        freqs = [220, 440]

        profiler.gonna('interp')
        for i, f0 in enumerate(freqs):
            harmonics[
                i * N_HARMONICS : (i+1) * N_HARMONICS
            ] = [
                # Harmonic(f, envelope(f)) if f < NYQUIST - FMAX else Harmonic(f, 0)
                Harmonic(f, envelope(f))
                for f in LADDER * f0
            ]
        for i in range(
            (i + 1) * N_HARMONICS, MAX_NOTES * N_HARMONICS
        ):  # same `i`. Cool shit
            harmonics[i] = Harmonic(harmonics[i].freq, 0) 

        profiler.gonna('eat')
        hSynth.eat(harmonics)

        profiler.gonna('mix')
        mixed = np.round(
            hSynth.mix() * INT32RANGE * MASTER_VOLUME
        ).astype(DTYPE_IO[0])
        streamOutContainer[0].write(mixed, PAGE_LEN)
        if WRITE_FILE is not None:
            f.writeframes(mixed)

        profiler.display(same_line=True)
        profiler.gonna('idle')
        return (None, pyaudio.paContinue)
    except:
        terminateLock.release()
        import traceback
        traceback.print_exc()
        return (None, pyaudio.paAbort)

main()
