import numpy as np
import librosa
import os
from glob import glob


class Time:
    def __init__(self, sampling_rate, n_fft, frame_num, number_file, min_sample, batch_size,
                 path_1=str, path_2=str):
        self.number_file = number_file
        self.frame_num = frame_num * 2
        self.n_fft = n_fft
        self.sampling_rate = sampling_rate
        self.min_sample = min_sample
        self.batch_size = batch_size
        self.padding = n_fft * frame_num
        self.y_data = None
        self.y_data_spectrum = None
        self.y_data_magnitude = None
        self.regularization = 0
        self.path_1 = path_1
        self.path_2 = path_2
        self.phase = None

        speaker_dir = [f.path for f in os.scandir(self.path_1) if f.is_dir()]

        chapter_dir = []
        for one_path in speaker_dir:
            chapter_dir += [f.path for f in os.scandir(one_path) if f.is_dir()]

        segment_name = []
        for one_path in chapter_dir:
            segment_name += glob(one_path + '\\*.flac')

        delete_file = []
        for one_path in segment_name:
            if os.stat(one_path).st_size < self.min_sample:
                delete_file.append(one_path)

        for one_path in delete_file:
            segment_name.remove(one_path)

        self.file_name = segment_name[:self.number_file]  ## 규격에 맞는 스피치 파일 저장

    def make_speech(self, file_number):
        wave, sr = librosa.load(self.file_name[file_number], sr=self.sampling_rate)
        if wave.shape[0] >= self.padding:
            wave = wave[:self.padding]
        else:
            wave = np.concatenate((wave, np.zeros(self.padding - wave.shape[0])), axis=0)  ## (160000)

        return wave

    def make_noise(self, noise_name: str):
        res_temp = None
        for i in range(16):
            if i < 9:
                noise_path = self.path_2 + noise_name + '\\ch0' + str(i + 1) + '.wav'
            else:
                noise_path = self.path_2 + noise_name + '\\ch' + str(i + 1) + '.wav'

            noise, sr = librosa.load(noise_path, sr=self.sampling_rate)
            if noise.shape[0] >= self.padding:
                noise = noise[2500:2500+self.padding]
            # noise = noise + 1e-10 # (160000)

            if i == 0:
                res_temp = np.copy(noise)
                res_temp = np.expand_dims(res_temp, axis=0)  # (1, 160000)
            else:
                noise = np.expand_dims(noise, axis=0)
                res_temp = np.concatenate((res_temp, noise), axis=0)

            res = np.copy(res_temp)
        for j in range(1, self.number_file // self.batch_size):  ## 노이즈 개수 조절
            res = np.concatenate((res, res_temp))

        return res

    def load_data(self, noise=None, noise_snr=5):
        if self.y_data is None:
            data = self.make_speech(0)
            data = np.expand_dims(data, axis=0)

            for i in range(1, self.number_file):
                temp = self.make_speech(i)
                temp = np.expand_dims(temp, axis=0)
                data = np.concatenate((data, temp), axis=0)

            self.y_data = np.copy(data)

        res = np.copy(self.y_data)  # (32, 160000)

        if noise is not None:
            for i in range(self.number_file):
                scale = adjust_snr(res[i], noise[i], noise_snr)
                res[i] += noise[i] * scale

            val_max = max(np.abs(np.max(res)), np.abs(np.min(res)))
            if val_max > self.regularization:
                self.regularization = val_max

        return res


class Spectrum(Time):
    def __init__(self, sampling_rate, n_fft, frame_num, number_file, min_sample, batch_size,
                 path_1=str, path_2=str):
        super().__init__(sampling_rate, n_fft, frame_num, number_file, min_sample, batch_size, path_1, path_2)

    def load_data(self, noise=None, noise_snr=5):
        if self.y_data is None:
            data = self.make_speech(0)
            data_spectrum = self.make_spectrum(data)
            data_spectrum = np.transpose(data_spectrum, (1, 0))
            data_spectrum = data_spectrum[:, 1:]
            data = np.expand_dims(data, axis=0)
            data_spectrum = np.expand_dims(data_spectrum, axis=0)  ## (1, 999, 160)

            for i in range(1, self.number_file):
                temp = self.make_speech(i)
                temp_spectrum = self.make_spectrum(temp)
                temp_spectrum = np.transpose(temp_spectrum, (1, 0))
                temp_spectrum = temp_spectrum[:, 1:]
                temp = np.expand_dims(temp, axis=0)
                temp_spectrum = np.expand_dims(temp_spectrum, axis=0)
                data = np.concatenate((data, temp), axis=0)  ##(32, 160000)
                data_spectrum = np.concatenate((data_spectrum, temp_spectrum), axis=0)  ##(32, 999 ,160)

            self.y_data = np.copy(data)
            self.y_data_spectrum = np.copy(data_spectrum)

        res = np.copy(self.y_data)
        res_spectrum = np.copy(self.y_data_spectrum)

        if noise is not None:
            for i in range(self.number_file):
                scale = adjust_snr(res[i], noise[i], noise_snr)
                res[i] += noise[i] * scale

            val_max = max(np.abs(np.max(res)), np.abs(np.min(res)))
            if val_max > self.regularization:
                self.regularization = val_max
            res_spectrum = self.make_spectrum(res)
            res_spectrum = np.transpose(res_spectrum, (0, 2, 1))
            res_spectrum = res_spectrum[:, :, 1:]

        return res_spectrum

    def make_spectrum(self, wave):
        spectrum = librosa.stft(wave, n_fft=self.n_fft, hop_length=self.n_fft // 2, win_length=self.n_fft,
                                window='hann', center=False)
        # spectrum = np.transpose(spectrum, (1, 0))
        # spectrum = spectrum[:, 1:]
        return spectrum


class spectral_magnitude(Time):
    def __init__(self, sampling_rate, n_fft, frame_num, number_file, min_sample, batch_size,
                 path_1=str, path_2=str):
        super().__init__(sampling_rate, n_fft, frame_num, number_file, min_sample, batch_size, path_1, path_2)

    def load_data(self, noise=None, noise_snr=5):
        if self.y_data is None:
            data = self.make_speech(0)
            data_magnitude = self.make_spectrum(data)
            data_magnitude = np.transpose(data_magnitude, (1, 0))
            data_magnitude = data_magnitude[:, 1:]
            data_magnitude = self.make_spectral_magnitude(data_magnitude)
            data = np.expand_dims(data, axis=0)
            data_magnitude = np.expand_dims(data_magnitude, axis=0)

            for i in range(1, self.number_file):
                temp = self.make_speech(i)
                temp_magnitude = self.make_spectrum(temp)
                temp_magnitude = np.transpose(temp_magnitude, (1,0))
                temp_magnitude = temp_magnitude[:, 1:]
                temp_magnitude = self.make_spectral_magnitude(temp_magnitude)
                temp = np.expand_dims(temp, axis=0)
                temp_magnitude = np.expand_dims(temp_magnitude, axis=0)
                data = np.concatenate((data, temp), axis=0)
                data_magnitude = np.concatenate((data_magnitude, temp_magnitude))

            self.y_data = np.copy(data)
            self.y_data_magnitude = np.copy(data_magnitude)

        res = np.copy(self.y_data)
        res_magnitude = np.copy(self.y_data_magnitude)

        if noise is not None:
            for i in range(self.number_file):
                scale = adjust_snr(res[i], noise[i], noise_snr)
                res[i] += noise[i] * scale

            val_max = max(np.abs(np.max(res)), np.abs(np.min(res)))
            if val_max > self.regularization:
                self.regularization = val_max
            res_magnitude = self.make_spectrum(res)
            res_magnitude = np.transpose(res_magnitude, (0, 2, 1))
            res_magnitude = res_magnitude[:, :, 1:]
            res_magnitude = self.make_spectral_magnitude(res_magnitude)

        return res_magnitude

    def make_spectrum(self, wave):
        spectrum = librosa.stft(wave, n_fft=self.n_fft, hop_length=self.n_fft // 2, win_length=self.n_fft,
                                window='hann', center=False)
        # spectrum = np.transpose(spectrum, (1, 0))
        # spectrum = spectrum[:, 1:]
        return spectrum

    def make_spectral_magnitude(self, spectrum, noise=None):
        magnitude, phase = librosa.magphase(spectrum)
        if phase.ndim == 2:
            phase = np.expand_dims(phase, axis=0)
            if noise is None:
                if self.phase is None:
                    self.phase = np.copy(phase)
                else:
                    self.phase = np.concatenate((self.phase, phase), axis=0)

        return magnitude


def adjust_snr(target, noise, db):  # Because of abs, it didn't return good scale value. We need bug fix
    sum_original = np.power(np.abs(target), 2)
    sum_noise = np.power(np.abs(noise), 2)
    sum_original = np.sum(sum_original)
    sum_noise = np.sum(sum_noise)
    sum_original = np.log10(sum_original)
    sum_noise = np.log10(sum_noise)
    scale = np.power(10, (sum_original-sum_noise)/2-(db/20))
    # SNR = 10 * log(power of signal(S)/power of noise(N))
    # SNR = 10 * (log(S) - log(N) - 2 log(noise scale))
    # log(noise scale) = (log(S) - log(N))/2 - SNR/20

    return scale
