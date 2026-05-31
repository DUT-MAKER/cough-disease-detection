import numpy as np
import h5py
import os
import random
import math
import torch
import pandas as pd
import librosa
from utils.config import valid_labels, RESP_CLASSES
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing


class JSTSP2021(Dataset):
    def __init__(self, hdf5_path, flusense):
        """Log mel spectrogram of the Dicova Task2 dataset."""
        self.hdf5_path = hdf5_path
        self.flusense = flusense
        self.le = preprocessing.LabelEncoder()
        self.le.fit(valid_labels)

    def __len__(self):
        hf = h5py.File(self.hdf5_path, 'r')
        res = hf['audio_name'].len()
        hf.close()
        return res

    def __getitem__(self, index):
        """Get input and target data of an audio clip.
        """
        with h5py.File(self.hdf5_path, 'r') as hf:
            audio_name = hf['audio_name'][index].decode()
            logmel = hf['logmel'][index]
            if self.flusense:
                label = hf['label'][index].decode()
                label = self.le.transform([label]).item()
            else:
                label = hf['label'][index]
        return {'audio_name': audio_name, 'logmel': logmel, 'label': label}


def dev_fold_dataset(dataset, split_ratio, shuffle, random_seed):
    """Return the train/val dataset for the given fold """
    indices = list(range(len(dataset)))
    data = DataLoader(dataset)
    labels = []
    for d in data:
        labels.append(d['label'].item())
    if len(indices) != len(labels):
        raise Exception('Incorrect train/val split!')

    random.seed(random_seed)
    X = random.sample(indices, k=len(dataset))
    y = labels
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split_ratio, random_state=random_seed,
                                                      shuffle=shuffle, stratify=y)

    train_indices = [i for i, e in enumerate(X) if e in X_train]
    val_indices = [i for i, e in enumerate(X) if e in X_val]
    if len(train_indices) + len(val_indices) != len(indices) or len(set(train_indices) & set(val_indices)) != 0:
        raise Exception('Incorrect train/val split!')
    '''
    split = int(np.floor(split * len(dataset)))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    '''
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    return train_dataset, val_dataset


def dicova_fold_dataset(dataset, list_dir, fold):
    """Create Dataloaders for dicova"""
    filenames = {}
    pos = 0
    # Iterate the indicated train and val fold
    for fun in ['train', 'val']:
        with open(os.path.join(list_dir, 'dicova_list', '{}_fold_{}.txt'.format(fun, fold))) as f:
            names = f.readlines()
        filenames[fun] = [x.strip() for x in names]

    # Draw from the Dataset according to the given train/val split
    train_indices = []
    val_indices = []
    for i in range(len(dataset)):
        if dataset[i]['audio_name'] in filenames['train']:
            train_indices.append(i)
            if dataset[i]['label'] == 1:
                pos += 1 
        if dataset[i]['audio_name'] in filenames['val']:
            val_indices.append(i)
    if len(train_indices) + len(val_indices) != len(dataset) or len(set(train_indices) & set(val_indices)) != 0:
        raise Exception('Incorrect train/val split!')

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    weights = [pos/len(train_indices), 1- pos/len(train_indices)]

    return train_dataset, val_dataset, weights


def split_compare_dataset(dataset):
    devel_indices = []
    test_indices = []
    train_indices = []

    pos = 0

    for idx in range(len(dataset)):
        if 'devel' in dataset[idx]['audio_name']:
            devel_indices.append(idx)
        elif 'train' in dataset[idx]['audio_name']:
            train_indices.append(idx)
            if dataset[idx]['label'] == 1:
                pos += 1
        else:
            test_indices.append(idx)

    devel_dataset = torch.utils.data.Subset(dataset, devel_indices)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    weights = [pos / len(train_indices), 1 - pos / len(train_indices)]

    return devel_dataset, train_dataset, test_dataset, weights


# ─── Respiratory Disease Dataset (Track A – CNN) ───────────────────────────
def _get_respiratory_label(row) -> int:
    """Map Sound-Dr CSV row → respiratory class index.
    Priority: COVID(4) > Pneumonia(3) > COPD(2) > Asthma(1) > Healthy(0)
    """
    cov  = str(row.get('cov19_status_choice', 'never')).strip().lower()
    cond = str(row.get('medical_condition_choice', "['No']'")).lower()
    if cov in ('last14', 'over14'):
        return 4
    if any(k in cond for k in ('pneumonia', 'lung', 'pulmonary', 'cystic')):
        return 3
    if 'copd' in cond:
        return 2
    if 'asthma' in cond:
        return 1
    return 0


class RespiratoryDataset(Dataset):
    """Reads Sound-Dr CSV + WAV files, returns Log-Mel Spectrogram + label.

    Args:
        data_csv  : path to sounddr_data/data.csv
        audio_root: root directory that contains cough/*.wav files
        sr        : target sample rate (default 16000)
        duration  : clip length in seconds (default 10)
        n_mels    : mel bins (default 64)
    """

    def __init__(self, data_csv: str, audio_root: str,
                 sr: int = 16000, duration: int = 10, n_mels: int = 64):
        self.audio_root = audio_root
        self.sr         = sr
        self.max_len    = sr * duration
        self.n_mels     = n_mels
        self.n_fft      = 512
        self.hop_length = 256

        df = pd.read_csv(data_csv)
        # Filter audio errors if column present
        if 'error' in df.columns:
            df = df[df['error'] == 0]
        df['file_path']          = df['file_cough'].astype(str) + '.wav'
        df['label_respiratory']  = df.apply(_get_respiratory_label, axis=1)
        self.df = df.reset_index(drop=True)

        # Class weights for CrossEntropyLoss (inverse frequency)
        counts = np.bincount(self.df['label_respiratory'].values,
                             minlength=len(RESP_CLASSES)).astype(float)
        total  = counts.sum()
        self.class_weights = [
            total / (len(RESP_CLASSES) * c) if c > 0 else 0.0
            for c in counts
        ]

    def __len__(self):
        return len(self.df)

    def _load_audio(self, path: str) -> np.ndarray:
        try:
            audio, _ = librosa.load(path, sr=self.sr, mono=True, res_type='kaiser_fast')
        except Exception:
            audio = np.zeros(self.max_len, dtype=np.float32)
        if len(audio) < self.max_len:
            n = math.ceil(self.max_len / len(audio))
            audio = np.tile(audio, n)
        return audio[:self.max_len].astype(np.float32)

    def _compute_logmel(self, audio: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=audio, sr=self.sr,
            n_fft=self.n_fft, hop_length=self.hop_length,
            n_mels=self.n_mels, fmax=8000
        )
        logmel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        return logmel  # shape: (n_mels, time_frames)

    def __getitem__(self, index):
        row       = self.df.iloc[index]
        audio_path = os.path.join(self.audio_root, row['file_path'])
        audio     = self._load_audio(audio_path)
        logmel    = self._compute_logmel(audio)  # (64, T)
        logmel    = logmel[np.newaxis, :]          # (1, 64, T) – channel dim
        label     = int(row['label_respiratory'])
        return {
            'audio_name': row['file_path'],
            'logmel'    : torch.tensor(logmel, dtype=torch.float32),
            'label'     : label,
        }
