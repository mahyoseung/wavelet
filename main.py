import HSSY_module
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, TimeDistributed, Concatenate, Conv1D, Dropout, LeakyReLU
from tensorflow.keras import Model
import pywt
import evaluate


def apply_soft_threshold(coeffs, threshold):
    return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)
## db1, db4, sym5, coif3


def denoise_wavelet_soft_threshold(transformed_data, threshold):
    denoised_data = []
    for sample_coeffs in transformed_data:
        denoised_sample_coeffs = [apply_soft_threshold(coeffs, threshold) for coeffs in sample_coeffs]
        denoised_data.append(denoised_sample_coeffs)
    return denoised_data


def apply_wavelet_transform(data, wavelet='db1', level=1):
    transformed_data = []
    # window = np.hanning(len(data[0]))

    for sample in data:
        # sample *= window
        coeffs = pywt.wavedec(sample, wavelet, level=level)
        transformed_data.append(coeffs)
    return transformed_data


def apply_inverse_wavelet_transform(transformed_data, wavelet='db1'):
    reconstructed_data = []
    # window = np.hanning(len(transformed_data[0]))

    for sample_coeffs in transformed_data:
        sample = pywt.waverec(sample_coeffs, wavelet)
        # sample *= window
        reconstructed_data.append(sample)
    return np.array(reconstructed_data)



noise_list = ['DKITCHEN', 'DWASHING', 'NFIELD', 'OOFFICE', 'DLIVING', 'NRIVER', 'SCAFE', 'NPARK']
batch_size = 16
number_batch = 2
lr = 1e-4
EPOCHS = 500
amount = batch_size * number_batch * len(noise_list) * 15 / 16
amount = int(amount)

data = HSSY_module.Time(sampling_rate=16000, n_fft=320, frame_num=500,
                                      number_file=batch_size * number_batch,
                                      min_sample=250000, batch_size=batch_size,
                                      path_1='..\\dataset\\LibriSpeech\\train-clean-100\\',
                                      path_2='..\\dataset\\demand\\')

noise_temp = data.make_noise(noise_list[0])
y_data = np.copy(data.load_data())
x_data = np.copy(data.load_data(noise_temp))

for i in range(1, len(noise_list)):
    noise_temp = data.make_noise(noise_list[i])
    x_data_temp = data.load_data(noise_temp)
    y_data_temp = data.load_data()
    x_data = np.concatenate((x_data, x_data_temp), axis=0)
    y_data = np.concatenate((y_data, y_data_temp), axis=0)

x_data /= data.regularization

x_data_test = x_data[amount:, :]
x_data = x_data[:amount, :]
y_data_test = y_data[amount:, :]
y_data = y_data[:amount, :]

x_data_wavelet = apply_wavelet_transform(x_data)
x_data_test_wavelet = apply_wavelet_transform(x_data_test)

soft_threshold = 0.03

x_data_denoised_soft = denoise_wavelet_soft_threshold(x_data_wavelet, soft_threshold)

x_data_test_denoised_soft = denoise_wavelet_soft_threshold(x_data_test_wavelet, soft_threshold)

x_data_denoised_inverse_soft = apply_inverse_wavelet_transform(x_data_denoised_soft)
x_data_test_denoised_inverse_soft = apply_inverse_wavelet_transform(x_data_test_denoised_soft)

evaluate.save_raw(x_data_test_denoised_inverse_soft, 1, "output.wav", 1)
evaluate.save_raw(y_data_test, 1, "origin.wav", 1)

train_dataset = tf.data.Dataset.from_tensor_slices((x_data_denoised_inverse_soft, y_data)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_data_test_denoised_inverse_soft, y_data_test)).batch(batch_size)


# class DenoiseGenerator(Model):
#     def __init__(self):
#         super(DenoiseGenerator, self).__init__()
#         self.conv1d_1 = Conv1D(8, 5, padding='same', activation='relu')
#         self.conv1d_2 = Conv1D(16, 5, padding='same', activation='relu')
#         self.conv1d_3 = Conv1D(32, 5, padding='same', activation='relu')
#         self.conv1d_4 = Conv1D(32, 5, padding='same', activation='relu')
#
#     def call(self, inputs):
#         x = tf.expand_dims(inputs, axis=-1)
#         x = self.conv1d_1(x)
#         x = self.conv1d_2(x)
#         x = self.conv1d_3(x)
#         x = self.conv1d_4(x)
#         x = tf.reduce_mean(x, axis=-1)
#         return x
#
#
# generator = DenoiseGenerator()
# generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
# loss_object = tf.keras.losses.MeanSquaredError()
# train_loss = tf.keras.metrics.Mean(name='train_loss')
# test_loss = tf.keras.metrics.Mean(name='test_loss')
#
#
# @tf.function
# def train_step(noisy_wave, original_wave):
#
#     with tf.GradientTape() as tape:
#         denoise_wave = generator(noisy_wave, training=True)
#         loss = loss_object(original_wave, denoise_wave)
#     gradients = tape.gradient(loss, generator.trainable_variables)
#     generator_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
#
#     train_loss(loss)
#
#
# @tf.function
# def test_step(noisy_wave, original_wave):
#
#     denoise_wave = generator(noisy_wave, training=False)
#     loss = loss_object(original_wave, denoise_wave)
#     test_loss(loss)
#     return denoise_wave
#
#
# def calculate_snr(original_audio, noise):
#     energy_original_audio = np.sum(np.square(original_audio))
#     energy_noise = np.sum(np.square(noise))
#     snr_original_audio = 10 * np.log10(energy_original_audio / energy_noise)
#     return snr_original_audio
#
#
# for epoch in range(EPOCHS):
#     start = time.time()
#     train_loss.reset_states()
#     test_loss.reset_states()
#     x_pred = None
#
#     for x_wave, y_wave in train_dataset:
#         train_step(x_wave, y_wave)
#
#     snr = 0.
#     for x_wave, y_wave in test_dataset:
#         denoise_wave = test_step(x_wave, y_wave)
#         snr += calculate_snr(y_wave, (denoise_wave - y_wave))/batch_size
#         if x_pred is None:
#             x_pred = np.copy(denoise_wave)
#         else:
#             x_pred = np.concatenate((x_pred, denoise_wave), axis=0)
#
#     evaluate.save_raw(x_pred, 1, "output.wav", 1)
#
#     print(
#         f'Epoch {epoch + 1}, '
#         f'Train_Loss: {train_loss.result()}, '
#         f'Test_Loss: {test_loss.result()}, '
#         f'Time: {time.time() - start} sec'
#     )
#     print(
#         f'SNR: {snr}db, '
#     )