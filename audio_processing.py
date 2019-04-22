import numpy as np
from python_speech_features import fbank, delta
import torch
try:
    from . import constants as c
except ValueError:
    import constants as c
import librosa
from librosa.feature import melspectrogram
from librosa.core import ifgram

import os
import random


def mk_MFB(filename, sample_rate=c.SAMPLE_RATE, use_delta = c.USE_DELTA, use_scale = c.USE_SCALE, use_logscale = c.USE_LOGSCALE, replace = True):
    if not replace and os.path.exists(filename.replace('.wav', '.npy')):
        return

    # Process
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    filter_banks, energies = fbank(audio, samplerate=sample_rate, nfilt=c.FILTER_BANK, winlen=0.025)

    if use_logscale:
        filter_banks = 20 * np.log10(np.maximum(filter_banks,1e-5))

    if use_delta:
        delta_1 = delta(filter_banks, N=1)
        delta_2 = delta(delta_1, N=1)

        filter_banks = normalize_frames(filter_banks, Scale=use_scale)
        delta_1 = normalize_frames(delta_1, Scale=use_scale)
        delta_2 = normalize_frames(delta_2, Scale=use_scale)

        frames_features = np.hstack([filter_banks, delta_1, delta_2])
    else:
        filter_banks = normalize_frames(filter_banks, Scale=use_scale)
        frames_features = filter_banks

    # Save
    np.save(filename.replace('.wav', '.npy'), frames_features)
    return


def mk_mel(filename, sample_rate=c.SAMPLE_RATE, replace = True):
    if not replace and os.path.exists(filename.replace('.wav', '.npy')):
        return

    # Process
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    gram = melspectrogram(y=audio, sr=sample_rate, n_mels=c.MEL_FEATURES).astype(np.float16)
    gram = gram.transpose()

    # Save
    np.save(filename.replace('.wav', '.npy'), gram)
    return


def mk_if(filename, sample_rate=c.SAMPLE_RATE, replace = True):
    if not replace and os.path.exists(filename.replace('.wav', '.npy')):
        return

    # Process
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    gram = ifgram(y=audio, sr=sample_rate, n_fft=(c.IF_FEATURES - 1) * 2)[0]
    gram = gram.transpose()
    print(gram.shape)

    # Save
    np.save(filename.replace('.wav', '.npy'), gram)
    return



def read_npy(filename):
    audio = np.load(filename.replace('.wav', '.npy'))
    return audio


def read_audio(filename, sample_rate=c.SAMPLE_RATE):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio.flatten()
    return audio


def normalize_frames(m,Scale=True):
    if Scale:
        return (m - np.mean(m, axis=0)) / (np.std(m, axis=0) + 2e-12)
    else:
        return (m - np.mean(m, axis=0))


class truncatedinput(object):
    """Rescales the input PIL.Image to the given 'size'.
    If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.
    If 'size' is a number, it will indicate the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, num_frames, truncate=True):

        super(truncatedinput, self).__init__()
        self.num_frames = num_frames
        self.truncate = truncate

    def __call__(self, frames_features):
  
        # Check if to slice
        if not self.truncate:
            return frames_features

        # Shapes
        shape = frames_features.shape
        num_frames = shape[-2]
        num_features = shape[-1]
        batch_size = shape[0]
        frames_features = frames_features.view((shape[-2], shape[-1], -1))

        import random
        if self.num_frames <= num_frames:
            j = random.randrange(0, num_frames - self.num_frames)
            frames_slice = frames_features[j:j + self.num_frames]
        else:
            frames_slice = torch.zeros([self.num_frames, num_features, batch_size], dtype=torch.float64)
            frames_slice[0:num_frames] = frames_features

        # Changed Dimenisions
        shape = list(shape)
        shape[-2] = self.num_frames
        frames_slice = frames_slice.view(shape)
        return frames_slice



class totensor(object):
    """Transform Data into the correct shape and type
    
    Return: Tensor Float: shape (1, Frame, Features) or (Batch Size, 1, Frame, Features)
    """
    def __init__(self, permute=True):
        self.permute = permute

    def __call__(self, img):
        """
        Args:
            img (PIL.Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if isinstance(img, np.ndarray):
            img = torch.FloatTensor(img)

        if self.permute:
            axis = [ii for ii in range(0, len(img.shape))]
            axis[-1] = len(img.shape) - 2
            axis[-2] = len(img.shape) - 1
            img = img.permute(tuple(axis))

        if len(img.shape) == 2:
            img = img.view(1, img.shape[-2], img.shape[-1])
        return img


class tonormal(object):


    def __init__(self):
        self.mean = 0.013987
        self.var = 1.008


    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient

        print(self.mean)
        self.mean+=1
        return tensor
