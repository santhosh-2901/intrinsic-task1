import torch
import pandas as pd
import json
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class TextMelDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.data['mel_spectrogram'] = self.data['mel_spectrogram'].apply(json.loads)
        all_text = "".join(self.data['normalized_text'])
        self.vocab = sorted(list(set(all_text)))
        self.char_to_int = {char: i for i, char in enumerate(self.vocab)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['normalized_text']
        mel = torch.tensor(self.data.iloc[idx]['mel_spectrogram']).float().view(-1, 80)
        text_sequence = torch.tensor([self.char_to_int[char] for char in text], dtype=torch.long)
        return text_sequence, mel

class TextMelCollate:
    def __call__(self, batch):
        text_sequences, mel_spectrograms = zip(*batch)
        text_padded = pad_sequence(text_sequences, batch_first=True, padding_value=0)
        mel_padded = pad_sequence(mel_spectrograms, batch_first=True, padding_value=0)
        return text_padded, mel_padded