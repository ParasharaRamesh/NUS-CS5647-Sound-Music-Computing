import math
import json
import torch
import librosa
import torchaudio
import numpy as np
from torchaudio import transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import time
from torch import float32
from hparams import Hparams
from utils import read_json, save_json, ls, jpath


def move_data_to_device(data, device):
    ret = []
    for i in data:
        if isinstance(i, torch.Tensor):
            ret.append(i.to(device))
    return ret


def get_data_loader(split, args, fns=None):
    dataset = MyDataset(
        dataset_root=args['dataset_root'],
        split=split,
        sampling_rate=args['sampling_rate'],
        annotation_path=args['annotation_path'],
        sample_length=args['sample_length'],
        frame_size=args['frame_size'],
        song_fns=fns,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args['batch_size'],
        num_workers=args['num_workers'],
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return data_loader


def collate_fn(batch):
    '''
    This function help to
    1. Group different components into separate tensors.
    2. Pad samples in the maximum length in the batch.
    '''

    inp = []
    onset = []
    offset = []
    octave = []
    pitch = []
    max_frame_num = 0
    for sample in batch:
        max_frame_num = max(max_frame_num, sample[0].shape[0], sample[1].shape[0], sample[2].shape[0],
                            sample[3].shape[0], sample[4].shape[0])

    for sample in batch:
        inp.append(
            torch.nn.functional.pad(sample[0], (0, 0, 0, max_frame_num - sample[0].shape[0]), mode='constant', value=0))
        onset.append(
            torch.nn.functional.pad(sample[1], (0, max_frame_num - sample[1].shape[0]), mode='constant', value=0))
        offset.append(
            torch.nn.functional.pad(sample[2], (0, max_frame_num - sample[2].shape[0]), mode='constant', value=0))
        octave.append(
            torch.nn.functional.pad(sample[3], (0, max_frame_num - sample[3].shape[0]), mode='constant', value=0))
        pitch.append(
            torch.nn.functional.pad(sample[4], (0, max_frame_num - sample[4].shape[0]), mode='constant', value=0))

    inp = torch.stack(inp)
    onset = torch.stack(onset)
    offset = torch.stack(offset)
    octave = torch.stack(octave)
    pitch = torch.stack(pitch)

    return inp, onset, offset, octave, pitch


class MyDataset(Dataset):
    def __init__(self, dataset_root, split, sampling_rate, annotation_path, sample_length, frame_size,
                 song_fns=None):
        '''
        This dataset return an audio clip in a specific duration in the training loop, with its "__getitem__" function.
        '''
        self.dataset_root = dataset_root
        self.split = split
        self.dataset_path = jpath(self.dataset_root, self.split)
        self.sampling_rate = sampling_rate
        self.annotation_path = annotation_path
        self.all_annotations = read_json(self.annotation_path)
        self.duration = {}
        if song_fns == None:
            self.song_fns = ls(self.dataset_path)
            self.song_fns.sort()
        else:
            self.song_fns = song_fns
        self.index = self.index_data(sample_length)
        self.sample_length = sample_length
        self.frame_size = frame_size
        self.frame_per_sec = int(1 / self.frame_size)

        # new constants
        self.hop_length = int(self.sampling_rate / self.frame_per_sec)
        self.one_hot_pitch_dims = 13
        self.one_hot_octave_dims = 5

        # new cacheable values
        self.waveform_cache = {}  # key: audio_fp value: original waveform
        self.mel_spectrogram_cache = {}  # key: audio_fp value: mel spectrogram
        self.onset_roll_cache = {}  # key: audio_fp value: onset
        self.offset_roll_cache = {}  # key: audio_fp value: offset
        self.pitch_roll_cache = {}  # key: audio_fp value: pitch
        self.octave_roll_cache = {}  # key: audio_fp value: octave

        # cacheEverything()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        '''
        Return spectrogram and 4 labels of an audio clip
        The audio filename and the start time of this sample is specified by "audio_fn" and "start_sec"
        '''
        audio_fn, start_sec = self.index[idx]
        end_sec = start_sec + self.sample_length
        duration = self.duration[audio_fn]
        audio_fp = jpath(self.dataset_path, audio_fn, 'Mixture.mp3')

        ''' YOUR CODE: Load audio from file, and compute mel spectrogram '''

        if audio_fp not in self.mel_spectrogram_cache.keys():
            waveform = self.load_audio(audio_fp)  # here also there is a waveform cache
            self.compute_mel_spectrogram(audio_fp, waveform)  # here the mel spectrogram is stored in the cache!

        spectrogram_clip = self.clip_mel_spectrogram(self.mel_spectrogram_cache[audio_fp], start_sec, end_sec)

        ''' YOUR CODE: Extract the desired clip, i.e., 5 sec of info, from both spectrogram and annotation '''
        onset_roll, offset_roll, octave_roll, pitch_roll = self.get_labels(audio_fp, self.all_annotations[audio_fn],
                                                                           duration)

        onset_clip = onset_roll[start_sec * self.frame_per_sec: end_sec * self.frame_per_sec]
        offset_clip = offset_roll[start_sec * self.frame_per_sec: end_sec * self.frame_per_sec]
        octave_clip = octave_roll[start_sec * self.frame_per_sec: end_sec * self.frame_per_sec]
        pitch_class_clip = pitch_roll[start_sec * self.frame_per_sec: end_sec * self.frame_per_sec]

        return spectrogram_clip, onset_clip, offset_clip, octave_clip, pitch_class_clip

    def clip_mel_spectrogram(self, mel_spectrogram, start_sec, end_sec):
        mel_start_frame = self.frame_per_sec * start_sec
        mel_end_frame = self.frame_per_sec * end_sec

        # mel spectrogram clip for 5 seconds
        spectrogram_clip = mel_spectrogram[:, mel_start_frame: mel_end_frame]

        # transpose it
        spectrogram_clip = np.transpose(spectrogram_clip)
        return spectrogram_clip

    def compute_mel_spectrogram(self, audio_fp, waveform):
        # create a mel spectrogram
        if audio_fp not in self.mel_spectrogram_cache:
            t0 = time.time()
            melSpecTransform = transforms.MelSpectrogram(
                sample_rate=self.sampling_rate,
                n_mels=256,
                hop_length=self.hop_length,
                n_fft=1024
            )
            mel_spectrogram = melSpecTransform(waveform)
            self.mel_spectrogram_cache[audio_fp] = mel_spectrogram

        return self.mel_spectrogram_cache[audio_fp]

    def load_audio(self, audio_fp):
        if audio_fp not in self.waveform_cache:
            t0 = time.time()
            orig_waveform, sr = torchaudio.load(audio_fp, normalize=True)
            t1 = time.time()
            waveform = torchaudio.functional.resample(orig_waveform, orig_freq=sr, new_freq=self.sampling_rate)
            t2 = time.time()
            mono_waveform = torch.mean(waveform, dim=0)
            self.waveform_cache[audio_fp] = mono_waveform
        return self.waveform_cache[audio_fp]

    def index_data(self, sample_length):
        '''
        Prepare the index for the dataset, i.e., the audio file name and starting time of each sample
        '''
        index = []
        for song_fn in self.song_fns:
            if song_fn.startswith('.'):  # Ignore any hidden file
                continue
            duration = self.all_annotations[song_fn][-1][1]
            num_seg = math.ceil(duration / sample_length)
            for i in range(num_seg):
                index.append([song_fn, i * sample_length])
            self.duration[song_fn] = duration
        return index

    def get_labels(self, audio_fp, annotation_data, duration):
        '''
        This function read annotation from file, and then convert annotation from note-level to frame-level
        Because we will be using frame-level labels in training.
        '''
        frame_num = math.ceil(duration * self.frame_per_sec)
        isInOnsetCache = audio_fp in self.onset_roll_cache
        isInOffsetCache = audio_fp in self.offset_roll_cache
        isInPitchCache = audio_fp in self.pitch_roll_cache
        isInOctaveCache = audio_fp in self.octave_roll_cache

        if not isInOnsetCache and not isInOffsetCache and not isInPitchCache and not isInOctaveCache:
            t0 = time.time()
            octave_roll = torch.zeros(size=(frame_num + 1,), dtype=torch.long)
            pitch_roll = torch.zeros(size=(frame_num + 1,), dtype=torch.long)
            onset_roll = torch.zeros(size=(frame_num + 1,), dtype=torch.long)
            offset_roll = torch.zeros(size=(frame_num + 1,), dtype=torch.long)

            ''' YOUR CODE: Create frame level label for a song to facilitate consequent computation '''
            ''' They are: onset roll, offset roll, octave roll, and pitch class roll '''
            ''' Each XX roll is a vector with integer elements, vector length equals to the number of frames of the song '''
            ''' Value range for onset, offset, octave, pitch class: [0,1], [0,1], [0,4], [0,12] '''
            ''' For onset and offset, 1 means there exists onset/offset in this frame '''
            ''' For octave and pitch class, 0 means illegal pitch or silence '''
            # looping over the annotation_data
            for annotation in annotation_data:
                start_time, end_time, pitch = annotation
                start_frame = int(self.frame_per_sec * start_time)
                end_frame = int(self.frame_per_sec * end_time)
                octave, pitch_in_octave = self.get_octave_and_pitch_class_from_pitch(pitch)
                if octave == 0 and pitch_in_octave == 0:
                    # skip in case its spurious
                    continue

                # now set onset and offset
                onset_roll[start_frame] = 1
                offset_roll[end_frame] = 1

                # now set the octave and pitch as one hot vectors, octave is one of 5 classes, and pitch is one of 13 classes
                octave_roll[start_frame:end_frame + 1] = octave
                pitch_roll[start_frame:end_frame + 1] = pitch_in_octave

            # cache it
            self.onset_roll_cache[audio_fp] = onset_roll
            self.offset_roll_cache[audio_fp] = offset_roll
            self.octave_roll_cache[audio_fp] = octave_roll
            self.pitch_roll_cache[audio_fp] = pitch_roll

        return self.onset_roll_cache[audio_fp], self.offset_roll_cache[audio_fp], self.octave_roll_cache[audio_fp], \
            self.pitch_roll_cache[audio_fp]

    def get_octave_and_pitch_class_from_pitch(self, pitch, note_start=36):
        '''
        Convert MIDI pitch number to octave and pitch_class
        pitch: int, range [36 (octave 0, pitch_class 0), 83 (octave 3, pitch 11)]
                pitch = 0 means silence
        return: octave, pitch_class.
                if no pitch or pitch out of range, output: 0, 0
        '''
        # note numbers ranging from C2 (36) to B5 (83)
        # octave class ranging from 0 to 4, 1~4 are valid octave class, 0 represent silence
        # pitch_class ranging from 0 to 12, pitch class 1 to 12: pitch C to B, pitch class 0: unknown class / silence
        if pitch == 0:
            return 4, 12

        t = pitch - note_start
        octave = t // 12
        pitch_class = t % 12

        if pitch < note_start or pitch > 83:
            return 0, 0
        else:
            return octave + 1, pitch_class + 1
