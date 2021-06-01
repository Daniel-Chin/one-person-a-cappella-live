print('importing...')
import pyaudio
from time import time, sleep
import numpy as np
from numpy.fft import rfft
import scipy
from scipy.interpolate import interp1d
from threading import Lock
import wave
import mido
try:
    from interactive import listen
    from yin import yin
    from streamProfiler import StreamProfiler
    from hybridSynth import HybridSynth, Harmonic
    from selectAudioDevice import selectAudioDevice
except ImportError as e:
    module_name = str(e).split('No module named ', 1)[1].strip().strip('"\'')
    print(f'Missing module {module_name}. Please download at')
    print(f'https://github.com/Daniel-Chin/Python_Lib/blob/master/{module_name}.py')
    input('Press Enter to quit...')
    raise e

print('Preparing...')
DEBUG_NO_MIDI = False
PAGE_LEN = 512
N_HARMONICS = 50    # NYQUIST / 200
HYBRID_QUALITY = 12
DO_PROFILE = True
# WRITE_FILE = None
WRITE_FILE = f'demo_{time()}.wav'

MASTER_VOLUME = .2
SR = 22050
NYQUIST = SR // 2
DTYPE_BUF = np.float32
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
NOTE_ON = 'note_on'
NOTE_OFF = 'note_off'
SR_OVER_PAGE_LEN = SR / PAGE_LEN

streamOutContainer = []
terminate_flag = 0
terminateLock = Lock()
profiler = StreamProfiler(PAGE_LEN / SR, DO_PROFILE)
notes = []
notes_changed = True
notesLock = Lock()
hySynth = None
harmonics = []

if DO_PROFILE:
    _print = print
    def print(*a, **k):
        _print()
        _print(*a, **k)

def sft(signal, freq_bin):
    # Slow Fourier Transform
    return np.abs(np.sum(signal * np.exp(IMAGINARY_LADDER * freq_bin))) / PAGE_LEN

def main():
    global terminate_flag, f, hySynth
    terminateLock.acquire()
    hySynth = HybridSynth(
        HYBRID_QUALITY, SR, PAGE_LEN, DTYPE_BUF, 
    )
    pa = pyaudio.PyAudio()
    # in_i, out_i = selectAudioDevice(pa)
    in_i, out_i = None, None
    streamOutContainer.append(pa.open(
        format = DTYPE_IO[1], channels = 1, rate = SR, 
        output = True, frames_per_buffer = PAGE_LEN,
        output_device_index = out_i, 
    ))
    if WRITE_FILE is not None:
        f = wave.open(WRITE_FILE, 'wb')
        f.setnchannels(1)
        f.setsampwidth(4)   # 32 / 8 = 4
        f.setframerate(SR)
    streamIn = pa.open(
        format = DTYPE_IO[1], channels = 1, rate = SR, 
        input = True, frames_per_buffer = PAGE_LEN,
        input_device_index = in_i, 
        stream_callback = onAudioIn, 
    )
    streamIn.start_stream()
    print('Press ESC to quit. ')
    if not DEBUG_NO_MIDI:
        midiPort = mido.open_input(callback = onMidiIn)
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
        if not DEBUG_NO_MIDI:
            midiPort.close()
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
    harmonics_a = np.zeros((harmonics_f.size, ))
    spectrum_2 = np.square(np.abs(rfft(signal * HANN)))
    SR_OVER_PAGE_LEN_OVER_f0 = SR_OVER_PAGE_LEN / f0
    for i, energy_2 in enumerate(spectrum_2):
        n = round(i * SR_OVER_PAGE_LEN_OVER_f0)
        if n >= harmonics_f.size:
            break
        harmonics_a[n] += energy_2
    harmonics_a = np.sqrt(harmonics_a) / PAGE_LEN
    harmonics_a[0] = harmonics_a[1]
    f = interp1d(harmonics_f, harmonics_a)
    # max_f = harmonics_f[-1]
    def envelope(x):
        # if x >= max_f:
        #     return 0
        try:
            return f(x)
        except ValueError:
            return 0
    return envelope

def onAudioIn(in_data, sample_count, *_):
    global notes_changed
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
        page = np.multiply(page, INV_INT32RANGE, dtype=DTYPE_BUF)

        profiler.gonna('getE')
        envelope = getEnvelope(page, PAGE_LEN)

        profiler.gonna('note_in')
        if DEBUG_NO_MIDI:
            if not notes:
                notes_changed = True
                notes.append(53)
        notesLock.acquire()
        if notes_changed:
            freqs = [pitch2freq(n) for n in notes]
            notes_changed = False
            notesLock.release()
            harmonics.clear()
            for f0 in freqs:
                for fn in LADDER * f0:
                    if fn >= NYQUIST:
                        break
                    harmonics.append(
                        Harmonic(fn, 0)
                    )
            harmonics.sort(key=Harmonic.getFreq)
        else:
            notesLock.release()

        profiler.gonna('interp')
        for harmonic in harmonics:
            harmonic.mag = envelope(harmonic.freq)
        
        profiler.gonna('eat')
        hySynth.eat(harmonics, skipSort=True)

        profiler.gonna('mix')
        if notes:
            mixed = hySynth.mix()
        else:
            mixed = page
        mixed = np.round(
            mixed * INT32RANGE * MASTER_VOLUME
        ).astype(DTYPE_IO[0])
        streamOutContainer[0].write(mixed, PAGE_LEN)
        if WRITE_FILE is not None:
            f.writeframes(mixed)

        profiler.display(same_line=False)
        profiler.gonna('idle')
        return (None, pyaudio.paContinue)
    except:
        terminateLock.release()
        import traceback
        traceback.print_exc()
        return (None, pyaudio.paAbort)

def pitch2freq(pitch):
    return np.exp((pitch + 36.37631656229591) * 0.0577622650466621)

def onMidiIn(msg):
    global notes_changed
    with notesLock:
        if msg.type == NOTE_ON:
            assert msg.note not in notes
            notes.append(msg.note)
        elif msg.type == NOTE_OFF:
            notes.remove(msg.note)
        notes_changed = True

main()
