import os
import numpy as np
import pickle
import glob
import librosa
from midi2audio import FluidSynth
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# --------------------------------------------- #
#                  Data Utils
# --------------------------------------------- #
SAMPLE_RATE = 22050


def midi_to_audio(midi_file):
    # Convert MIDI to WAV using FluidSynth
    fs = FluidSynth()
    audio_file = 'temp.wav'
    fs.midi_to_audio(midi_file, audio_file)

    # Load the WAV file using librosa
    y, _ = librosa.load(audio_file, sr=SAMPLE_RATE)

    return y


def audio_to_spectrograms(audio, target_shape=512):
    """
    Receives a audio and splits it into spectrograms of size target_shape
    """
    # Compute the spectrogram and convert to dB scale
    D = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_fft=2048, hop_length=1024,
                                       n_mels=256)
    D = librosa.power_to_db(D, ref=np.max)

    # Split spectrogram into chunks of size target_shape
    chunks = []
    for i in range(0, D.shape[1], target_shape):
        chunk = D[:, i:i + target_shape]
        if chunk.shape[1] == target_shape:
            chunks.append(chunk)

    return chunks


def midi_to_spectrograms(midi_file, target_shape=512):
    """
    Receives a MIDI file and splits it into spectrograms of size target_shape
    """
    # Convert MIDI to audio
    y = midi_to_audio(midi_file)

    # Convert audio to spectrograms
    chunks = audio_to_spectrograms(y, target_shape)

    return np.array(chunks)


def create_spectrogram_dataset(pickle_file_path):
    # Check if the pickle file exists
    if os.path.exists(pickle_file_path):
        # Load the spectrograms from the pickle file
        with open(pickle_file_path, 'rb') as f:
            spectrograms = pickle.load(f)
            print('Loaded spectrograms from pickle file.')
        return spectrograms

    raise "NotFound"


class MelSpectrograms(Dataset):
    def __init__(self, path):
        super().__init__()

        spectrograms = create_spectrogram_dataset(path)
        # Normalize between -1 and 1
        self.min_val = np.min(spectrograms)
        self.max_val = np.max(spectrograms)
        self._length = len(spectrograms)
        spectrograms = np.expand_dims(spectrograms, axis=1)
        self.images = (spectrograms - self.min_val) / \
            ((self.max_val - self.min_val) / 2) - 1

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        return self.images[i]


def load_data(args):
    train_data = MelSpectrograms(args.dataset_path)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False)
    return train_loader


# --------------------------------------------- #
#                  Module Utils
#            for Encoder, Decoder etc.
# --------------------------------------------- #

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def plot_images(images):
    # To do: View what is input to this function and adapt the code to plot it
    print(images)

    """x = images["input"]
    reconstruction = images["rec"]
    half_sample = images["half_sample"]
    full_sample = images["full_sample"]

    fig, axarr = plt.subplots(1, 4)
    axarr[0].imshow(x.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[1].imshow(reconstruction.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[2].imshow(half_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[3].imshow(full_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    plt.show()
    """
