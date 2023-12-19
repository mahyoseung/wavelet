import matplotlib.pyplot as plt
import numpy as np
import librosa
import soundfile as sf

def save_raw(wave, scale, file_name, file_number):
    res = wave[file_number]
    sf.write(file_name, res * scale, samplerate=16000)
