import os
import numpy as np
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy import signal
from glob import glob
import re
import pandas as pd
import gc
from scipy.io import wavfile
import librosa
import cv2

from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D, Conv2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GRU,MaxPooling1D,Activation,TimeDistributed, MaxPooling1D, Embedding, GlobalMaxPooling2D, GlobalMaxPooling1D
from sklearn.model_selection import train_test_split
import keras

L = 16000
legal_labels = 'yes no up down left right on off stop go silence unknown'.split()

#src folders
root_path = r'/home/gasimov_aydin/Speech/'
out_path = r'/home/gasimov_aydin/Speech/'
model_path = r'/home/gasimov_aydin/Speech/'
train_data_path = os.path.join(root_path, 'train', 'audio')
test_data_path = os.path.join(root_path, 'test', 'audio')

def custom_fft(y, fs):
    T = 1.0 / fs
    N = y.shape[0]
    yf = fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    # FFT is simmetrical, so we take just the first half
    # FFT is also complex, to we take just the real part (abs)
    vals = 2.0/N * np.abs(yf[0:N//2])
    return xf, vals

checkpointer = ModelCheckpoint(filepath='/home/gasimov_aydin/Speech/weights.hdf5', verbose=1, save_best_only=True)

def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)


def log_specgram3(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    D = librosa.stft(audio, n_fft=480, hop_length=160,
                     win_length=480, window='hamming')
    spect, phase = librosa.magphase(D)
    return 1, 1, spect

def list_wavs_fname(dirpath, ext='wav'):
    print(dirpath)
    fpaths = glob(os.path.join(dirpath, r'*/*' + ext))
    pat = r'.+/(\w+)/\w+\.' + ext + '$'
    labels = []
    for fpath in fpaths:
        r = re.match(pat, fpath)
        if r:
            labels.append(r.group(1))
    pat = r'.+/(\w+\.' + ext + ')$'
    fnames = []
    for fpath in fpaths:
        r = re.match(pat, fpath)
        if r:
            fnames.append(r.group(1))
    return labels, fnames

def pad_audio(samples):
    if len(samples) >= L: return samples
    else: return np.pad(samples, pad_width=(L - len(samples), 0), mode='constant', constant_values=(0, 0))

def time_shift(samples):
    start_ = int(np.random.uniform(-4800,4800))
    if start_ >= 0:
       wav_time_shift = np.r_[samples[start_:], np.random.uniform(-0.001,0.001, start_)]
    else:
       wav_time_shift = np.r_[np.random.uniform(-0.001,0.001, -start_), samples[:start_]]
    return wav_time_shift 

def add_noise(sapmles):
    bg_files = os.listdir('/home/gasimov_aydin/Speech/train/audio/_background_noise_/')
    bg_files.remove('README.md')
    chosen_bg_file = bg_files[np.random.randint(6)]
    bg, sr = librosa.load('/home/gasimov_aydin/Speech/train/audio/_background_noise_/'+chosen_bg_file, sr=None)

    start_ = np.random.randint(bg.shape[0]) #-16000
    bg_slice = bg[start_ : start_ + 16000]
    bg_slice2 = signal.resample(sapmles, int(new_sample_rate / len(bg_slice) * bg_slice.shape[0]))
    wav_with_bg = sapmles * np.random.uniform(0.8, 1.2) + \
              bg_slice2 * np.random.uniform(0, 0.1)
    return wav_with_bg

def add_simple_noise(samples):
    wn = np.random.randn(len(samples))
    data_wn = samples + 0.005*wn
    return data_wn

def stretch(data, rate=1.2):
    input_length = 16000
    data = librosa.effects.time_stretch(data, rate)
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, len(data) - input_length )), "constant")

    return data

def change_speed(samples):
    speed_rate = np.random.uniform(0.7,1.3)
    wav_speed_tune = cv2.resize(samples, (1, int(len(samples) * speed_rate))).squeeze()

    if len(wav_speed_tune) < 16000:
        pad_len = 16000 - len(wav_speed_tune)
        wav_speed_tune = np.r_[np.random.uniform(-0.001,0.001,int(pad_len/2)),
                           wav_speed_tune,
                           np.random.uniform(-0.001,0.001,int(np.ceil(pad_len/2)))]
    else:
        cut_len = len(wav_speed_tune) - 16000
        wav_speed_tune = wav_speed_tune[int(cut_len/2):int(cut_len/2)+16000]
    return wav_speed_tune


def chop_audio(samples, L=16000, num=100):
    for i in range(num):
        beg = np.random.randint(0, len(samples) - L)
        yield samples[beg: beg + L]

def label_transform(labels):
    nlabels = []
    for label in labels:
        if label == '_background_noise_':
            nlabels.append('silence')
       # elif label == '_silence_':
       #     nlabels.append('silence')   
       # elif label == '_unknown_':
       #     nlabels.append('unknown')           
        elif label not in legal_labels:
            nlabels.append('unknown')
        else:
            nlabels.append(label)
    return pd.get_dummies(pd.Series(nlabels))

labels, fnames = list_wavs_fname(train_data_path)

new_sample_rate = 16000
y_train = []
x_train = []

for label, fname in zip(labels, fnames):
    sample_rate, samples = wavfile.read(os.path.join(train_data_path, label, fname))
    samples = pad_audio(samples)
    time_samples = time_shift(samples)
    #noise_samples = stretch(samples,0.8) #add_noise(samples)
    if len(samples) > 16000:
        n_samples = chop_audio(samples)
    else: n_samples = [samples]
    for samples in n_samples:
        resampled = signal.resample(samples, int(new_sample_rate / sample_rate * samples.shape[0]))
        _, _, specgram = log_specgram(resampled, sample_rate=new_sample_rate)
        #_, _, specgram = log_specgram(samples, sample_rate=new_sample_rate)
        y_train.append(label)
        x_train.append(specgram)
  # --------------------------
    if len(time_samples ) > 16000:
        n_samples2 = chop_audio(time_samples )
    else: n_samples2 = [time_samples]
    for time_samples in n_samples2:
        resampled2 = signal.resample(time_samples , int(new_sample_rate / sample_rate * time_samples.shape[0]))
        _, _, specgram2 = log_specgram(resampled2, sample_rate=new_sample_rate)
        #_, _, specgram = log_specgram(samples, sample_rate=new_sample_rate)
        y_train.append(label)
        x_train.append(specgram2)
    # --------------------------
    #if len(noise_samples) > 16000:
    #    n_samples3 = chop_audio(noise_samples)
    #else: n_samples3 = [noise_samples]
    #for noise_samples in n_samples3:
    #    resampled3 = signal.resample(noise_samples , int(new_sample_rate / sample_rate * noise_samples .shape[0]))
    #    _, _, specgram3 = log_specgram(resampled3, sample_rate=new_sample_rate)
           #_, _, specgram = log_specgram(samples, sample_rate=new_sample_rate)
    #    y_train.append(label)
    #    x_train.append(specgram3)
x_train = np.array(x_train)
print('X_train before: ',x_train.shape)
x_train = x_train.reshape(tuple(list(x_train.shape)))
print('X_train after: ',x_train.shape)
y_train = label_transform(y_train)
label_index = y_train.columns.values
y_train = y_train.values
y_train = np.array(y_train)
del labels, fnames
gc.collect()

input_shape = (99, 161)
nclass = 12
l_drop_out = 0.1
x = Input(shape=input_shape)

conv_1d = Conv1D(8*(2 ** 1), (3),padding = 'same')(x)
conv_1d = Conv1D(8*(2 ** 1), (3),padding = 'same')(conv_1d)
conv_1d = BatchNormalization()(conv_1d)
conv_1d = Activation('relu')(conv_1d)
conv_1d = MaxPooling1D((2), padding='same')(conv_1d)
#conv_1d = Dropout(l_drop_out)(conv_1d)

conv_1d = Conv1D(8*(2 ** 2), (3),padding = 'same')(conv_1d)
conv_1d = Conv1D(8*(2 ** 2), (3),padding = 'same')(conv_1d)
conv_1d = BatchNormalization()(conv_1d)
conv_1d = Activation('relu')(conv_1d)
conv_1d = MaxPooling1D((2), padding='same')(conv_1d)
#conv_1d = Dropout(l_drop_out)(conv_1d)

conv_1d = Conv1D(8*(2 ** 3), (3),padding = 'same')(conv_1d)
conv_1d = Conv1D(8*(2 ** 3), (3),padding = 'same')(conv_1d)
conv_1d = BatchNormalization()(conv_1d)
conv_1d = Activation('relu')(conv_1d)
conv_1d = MaxPooling1D((2), padding='same')(conv_1d)
#conv_1d = Dropout(l_drop_out)(conv_1d)

conv_1d = Conv1D(8*(2 ** 4), (3),padding = 'same')(conv_1d)
conv_1d = Conv1D(8*(2 ** 4), (3),padding = 'same')(conv_1d)
conv_1d = BatchNormalization()(conv_1d)
conv_1d = Activation('relu')(conv_1d)
conv_1d = MaxPooling1D((2), padding='same')(conv_1d)
#conv_1d = Dropout(l_drop_out)(conv_1d)

conv_1d = Conv1D(8*(2 ** 5), (3),padding = 'same')(conv_1d)
conv_1d = Conv1D(8*(2 ** 5), (3),padding = 'same')(conv_1d)
conv_1d = BatchNormalization()(conv_1d)
conv_1d = Activation('relu')(conv_1d)
conv_1d = MaxPooling1D((2), padding='same')(conv_1d)
#conv_1d = Dropout(l_drop_out)(conv_1d)

conv_1d = Conv1D(8*(2 ** 6), (3),padding = 'same')(conv_1d)
conv_1d = Conv1D(8*(2 ** 6), (3),padding = 'same')(conv_1d)
conv_1d = BatchNormalization()(conv_1d)
conv_1d = Activation('relu')(conv_1d)
conv_1d = MaxPooling1D((2), padding='same')(conv_1d)
#conv_1d = Dropout(l_drop_out)(conv_1d)

#conv_1d = Conv1D(8*(2 ** 7), (3),padding = 'same')(conv_1d)
#conv_1d = Conv1D(8*(2 ** 7), (3),padding = 'same')(conv_1d)
#conv_1d = BatchNormalization()(conv_1d)
#conv_1d = Activation('relu')(conv_1d)
#conv_1d = MaxPooling1D((2), padding='same')(conv_1d)
#conv_1d = Dropout(l_drop_out)(conv_1d)

#conv_1d = Conv1D(8*(2 ** 8), (3),padding = 'same')(conv_1d)
#conv_1d = Conv1D(8*(2 ** 8), (3),padding = 'same')(conv_1d)
#conv_1d = BatchNormalization()(conv_1d)
#conv_1d = Activation('relu')(conv_1d)
#conv_1d = MaxPooling1D((2), padding='same')(conv_1d)
#conv_1d = Dropout(l_drop_out)(conv_1d)

#conv_1d = Conv1D(16, 3, 
#                     activation='relu',
#                     name='conv5')(conv_1d)
#conv_1d = Dropout(l_drop_out)(conv_1d)

#bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
#simp_rnn = GRU(161, activation='relu',
#        return_sequences=True, implementation=2, name='rnn1')(conv2d)
#simp_rnn = Dropout(l_drop_out)(simp_rnn)
#simp_rnn = GRU(161, activation='relu',
#        return_sequences=True, implementation=2, name='rnn2')(simp_rnn)
#simp_rnn = Dropout(l_drop_out)(simp_rnn)
#simp_rnn = GRU(81, activation='relu',
#        return_sequences=True, implementation=2, name='rnn3')(simp_rnn)
#simp_rnn = Dropout(l_drop_out)(simp_rnn)
#simp_rnn = GRU(81, activation='relu',
#        return_sequences=True, implementation=2, name='rnn4')(simp_rnn)
#simp_rnn = Dropout(l_drop_out)(simp_rnn)

img_1 = Flatten()(conv_1d)

#img_1 = GlobalMaxPooling1D()(conv_1d)

#dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(img_1))
#dense_1 = BatchNormalization()(Dense(64, activation=activations.relu)(dense_1))

dense_1 = BatchNormalization()(Dense(64, activation=activations.relu)(img_1))
#dense_1 = BatchNormalization()(Dense(256, activation=activations.relu)(dense_1))
#dense_1 = BatchNormalization()(Dense(64, activation=activations.relu)(dense_1))
dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)


model = models.Model(inputs=x, outputs=dense_1)

#model.load_weights('weights2.hdf5')

opt = optimizers.Adam()

model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=["accuracy"])
model.summary()

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=100)

model.fit(x_train, y_train, batch_size=128, validation_data=(x_valid, y_valid), epochs=30, shuffle=True, verbose=1, callbacks=[checkpointer])

#model.save(os.path.join(model_path, 'cnn.model'))


def test_data_generator(batch=128):
    fpaths = glob(os.path.join(test_data_path, '*wav'))
    i = 0
    for path in fpaths:
        if i == 0:
            imgs = []
            fnames = []
        i += 1
        rate, samples = wavfile.read(path)
        samples = pad_audio(samples)
        resampled = signal.resample(samples, int(new_sample_rate / rate * samples.shape[0]))
        _, _, specgram = log_specgram(resampled, sample_rate=new_sample_rate)
        imgs.append(specgram)
        fnames.append(path.split('\\')[-1])
        if i == batch:
            i = 0
            imgs = np.array(imgs)
            imgs = imgs.reshape(tuple(list(imgs.shape)))
            yield fnames, imgs
    if i < batch:
        imgs = np.array(imgs)
        imgs = imgs.reshape(tuple(list(imgs.shape)))
        yield fnames, imgs
    raise StopIteration()

del x_train, y_train
gc.collect()

index = []
results = []
for fnames, imgs in test_data_generator(batch=128):
    predicts = model.predict(imgs)
    predicts = np.argmax(predicts, axis=1)
    predicts = [label_index[p] for p in predicts]
#    fnames = fnames.replace("","")
    index.extend(fnames)
    results.extend(predicts)

df = pd.DataFrame(columns=['fname', 'label'])
df['fname'] = index
df['label'] = results
df.to_csv(os.path.join(out_path, 'sub.csv'), index=False)

