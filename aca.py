print('importing...')
import pyaudio
from time import time, sleep
import numpy as np
from numpy.fft import rfft
import scipy
from scipy.interpolate import interp1d
from threading import Lock
from scipy.stats import norm
from scipy.signal import butter, sosfiltfilt
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
FREEZE_COOLDOWN = 25
KEY_DELAY = .1
USE_KEY_VELOCITY = False
UNVOICE_USING_NOISE = True
HAS_UNPITCHED = True
BUTTER_ORDER = 1
DEBUG_NO_MIDI = False
PAGE_LEN = 768
N_HARMONICS = 50    # NYQUIST / 200
HYBRID_QUALITY = 60
DO_PROFILE = True
# WRITE_FILE = None
WRITE_FILE = f'demo_{time()}.wav'

MASTER_VOLUME = .2
SR = 22050
HYBRID_VERBOSE = False
NYQUIST = SR // 2
DTYPE_BUF = np.float32
DTYPE_IO = (np.int32, pyaudio.paInt32)
TWO_PI = np.pi * 2
HANN = scipy.signal.get_window('hann', PAGE_LEN, True)
SILENCE = np.zeros((PAGE_LEN, ))
IMAGINARY_LADDER = np.linspace(0, TWO_PI * 1j, PAGE_LEN)
LADDER = np.arange(1, N_HARMONICS + 1)
FMAX = 1600
FMIN = 120
INT32RANGE = 2**31
INV_INT32RANGE = 1 / 2**31
NOTE_ON = 'note_on'
NOTE_OFF = 'note_off'
CONTROL_CHANGE = 'control_change'
PAGE_LEN_OVER_SR = PAGE_LEN / SR
PAGE_TIME = PAGE_LEN_OVER_SR
SOS = butter(
    BUTTER_ORDER, 2000000 / SR / PAGE_LEN, btype='low', 
    output='sos', 
)
SPECTRUM_SIZE = PAGE_LEN // 2 + 1

streamOutContainer = []
terminate_flag = 0
terminateLock = Lock()
profiler = StreamProfiler(PAGE_LEN_OVER_SR, DO_PROFILE)
hySynth = None
ampedHarmonics = []
midiHandler = None

if DO_PROFILE:
    _print = print
    def print(*a, **k):
        _print()
        _print(*a, **k)

class AmpedHarmonic(Harmonic):
    __slot__ = ['freq', 'mag', 'amp']

    def __init__(self, freq, mag, amp):
        super().__init__(freq, mag)
        self.amp = amp

def main():
    global terminate_flag, f, hySynth, midiHandler
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
        midiHandler = MidiHandler()
        midiPort = mido.open_input(
            callback = midiHandler.onMidiIn, 
        )
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

def getEnvelope(signal, len_signal, spectrum, spectrum_complex):
    f0 = yin(signal, SR, len_signal, fmin=FMIN, fmax=FMAX)
    harmonics_f = np.arange(0, NYQUIST, f0)
    harmonics_a = np.zeros((harmonics_f.size, ))
    spectrum_2 = np.square(spectrum)
    for n, fn in enumerate(harmonics_f):
        mid_f_bin = round(fn * PAGE_LEN_OVER_SR)
        if mid_f_bin + 2 >= SPECTRUM_SIZE:
            break
        harmonics_a[n] += spectrum_2[mid_f_bin - 2]
        harmonics_a[n] += spectrum_2[mid_f_bin - 1]
        harmonics_a[n] += spectrum_2[mid_f_bin]
        harmonics_a[n] += spectrum_2[mid_f_bin + 1]
        harmonics_a[n] += spectrum_2[mid_f_bin + 2]
        spectrum[mid_f_bin - 1] = spectrum[mid_f_bin - 2]
        spectrum[mid_f_bin    ] = spectrum[mid_f_bin - 3]
        spectrum[mid_f_bin + 1] = spectrum[mid_f_bin - 4]
        spectrum_complex[mid_f_bin - 1] = 0
        spectrum_complex[mid_f_bin] = 0
        spectrum_complex[mid_f_bin + 1] = 0
        # spectrum_complex[mid_f_bin + 2] = 0
        # spectrum_complex[mid_f_bin - 2] = 0
    harmonics_a = np.sqrt(harmonics_a)
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
        midiHandler.delayKeys()

        profiler.gonna('rfft')
        spectrum_complex = rfft(page * HANN) / PAGE_LEN
        spectrum = np.abs(spectrum_complex)

        profiler.gonna('getE')
        envelope = getEnvelope(
            page, PAGE_LEN, spectrum, spectrum_complex, 
        )

        profiler.gonna('note_in')
        if DEBUG_NO_MIDI:
            if not midiHandler.notes:
                midiHandler.notes_changed = True
                midiHandler.notes[53] = .5
        if midiHandler.notes_changed:
            ampedHarmonics.clear()
            for pitch, amp in midiHandler.notes.items():
                f0 = pitch2freq(pitch)
                for fn in LADDER * f0:
                    if fn >= NYQUIST:
                        break
                    ampedHarmonics.append(
                        AmpedHarmonic(fn, 0, amp)
                    )
            midiHandler.notes_changed = False
            ampedHarmonics.sort(key=AmpedHarmonic.getFreq)

        profiler.gonna('interp')
        for aH in ampedHarmonics:
            aH.mag = envelope(aH.freq) * aH.amp
        
        if HAS_UNPITCHED:
            profiler.gonna('unvoic')
            if UNVOICE_USING_NOISE:
                unvoiced_envelope = sosfiltfilt(SOS, spectrum)
                random_spectrum = norm.rvs(
                    0, 1, SPECTRUM_SIZE, 
                ) + norm.rvs(0, 1j, SPECTRUM_SIZE)
                unvoiced_spectrum = random_spectrum * unvoiced_envelope
            else:
                unvoiced_spectrum = spectrum_complex
        else:
            unvoiced_spectrum = None

        profiler.gonna('eat')
        hySynth.eat(
            ampedHarmonics, unvoiced_spectrum * .3, 
            skipSort=True, verbose=HYBRID_VERBOSE
            # [], unvoiced_spectrum, skipSort=True, 
        )

        profiler.gonna('mix')
        if midiHandler.notes:
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

class MidiHandler:
    def __init__(self):
        self.midiQueueLock = Lock()
        self.midiDelayQueue = []
        self.notes_changed = True
        self._notes = {}        # internal true notes
        self.notes = self._notes    # external, can be frozen
        self.freeze_until = 0
    
    def onMidiIn(self, msg):
        with self.midiQueueLock:
            self.midiDelayQueue.append(
                (msg, time() + KEY_DELAY)
            )

    def delayKeys(self):
        while self.midiDelayQueue:
            with self.midiQueueLock:
                msg, sched_time = self.midiDelayQueue[0]
                if sched_time > time():
                    return
                self.midiDelayQueue.pop(0)
            self.handleMessage(msg)

    def handleMessage(self, msg):
        if msg.type == NOTE_ON:
            if USE_KEY_VELOCITY:
                self._notes[msg.note] = (msg.velocity ** 2) * .0001
            else:
                self._notes[msg.note] = .5
        elif msg.type == NOTE_OFF:
            if msg.note in self.notes:
                self._notes.pop(msg.note)
        elif msg.type == CONTROL_CHANGE:
            if msg.channel != 0:
                return
            if msg.value == 127:
                # press down
                if time() >= self.freeze_until:
                    print('freeze!')
                    self.notes = self._notes.copy()
                    print(self.notes)
                    self.freeze_until = time() + FREEZE_COOLDOWN
                else:
                    print('thru!')
                    self.notes = {}
            else:
                # release
                if time() >= self.freeze_until:
                    print('should not happen. 352786231')
                else:
                    self.notes = self._notes
        self.notes_changed = True

main()
