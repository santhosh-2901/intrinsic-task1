import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """Encodes the input text into a sequence of hidden states."""
    def __init__(self, n_vocab, embedding_dim=512, hidden_dim=256):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(n_vocab, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=3,
                            batch_first=True, bidirectional=True, dropout=0.1)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, _ = self.lstm(embedded)
        return outputs

class Attention(nn.Module):
    """Calculates attention weights."""
    def __init__(self, query_dim, key_dim):
        super(Attention, self).__init__()
        self.query_layer = nn.Linear(query_dim, query_dim, bias=False)
        self.key_layer = nn.Linear(key_dim, query_dim, bias=False)
        self.energy_layer = nn.Linear(query_dim, 1, bias=False)

    def forward(self, query, keys):
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_keys = self.key_layer(keys)
        
        energy = torch.tanh(processed_query + processed_keys)
        scores = self.energy_layer(energy).squeeze(-1)
        
        return F.softmax(scores, dim=-1)

class Decoder(nn.Module):
    """Decodes the encoder outputs into a mel spectrogram."""
    def __init__(self, encoder_dim, decoder_dim=1024, mel_dim=80):
        super(Decoder, self).__init__()
        self.attention_rnn = nn.LSTMCell(encoder_dim + mel_dim, decoder_dim)
        self.attention = Attention(decoder_dim, encoder_dim)
        self.decoder_rnn = nn.LSTMCell(decoder_dim + encoder_dim, decoder_dim)
        self.mel_projection = nn.Linear(decoder_dim, mel_dim)

    def forward(self, encoder_outputs, target_mels):
        batch_size = encoder_outputs.size(0)
        seq_len = target_mels.size(1)
        
        # --- FIX: Get the device from the input tensor ---
        device = encoder_outputs.device

        # --- FIX: Initialize all new tensors on the correct device ---
        attn_hidden = torch.zeros(batch_size, 1024, device=device)
        attn_cell = torch.zeros(batch_size, 1024, device=device)
        decoder_hidden = torch.zeros(batch_size, 1024, device=device)
        decoder_cell = torch.zeros(batch_size, 1024, device=device)
        prev_mel = torch.zeros(batch_size, 80, device=device)
        
        mel_outputs = []
        
        # Teacher-forcing loop
        for i in range(seq_len):
            attn_rnn_input = torch.cat((prev_mel, encoder_outputs[:, -1, :]), dim=-1)
            attn_hidden, attn_cell = self.attention_rnn(attn_rnn_input, (attn_hidden, attn_cell))

            attn_weights = self.attention(attn_hidden, encoder_outputs)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

            decoder_rnn_input = torch.cat((attn_hidden, context), dim=-1)
            decoder_hidden, decoder_cell = self.decoder_rnn(decoder_rnn_input, (decoder_hidden, decoder_cell))

            mel_output = self.mel_projection(decoder_hidden)
            mel_outputs.append(mel_output.unsqueeze(1))
            prev_mel = target_mels[:, i, :] 

        return torch.cat(mel_outputs, dim=1)

class Tacotron2(nn.Module):
    """The complete sequence-to-sequence model."""
    def __init__(self, n_vocab, mel_dim=80):
        super(Tacotron2, self).__init__()
        self.encoder = Encoder(n_vocab)
        self.decoder = Decoder(encoder_dim=512, mel_dim=mel_dim)
        self.postnet = nn.Sequential(
            nn.Conv1d(mel_dim, 512, kernel_size=5, padding=2), nn.BatchNorm1d(512), nn.Tanh(), nn.Dropout(0.5),
            nn.Conv1d(512, 512, kernel_size=5, padding=2), nn.BatchNorm1d(512), nn.Tanh(), nn.Dropout(0.5),
            nn.Conv1d(512, mel_dim, kernel_size=5, padding=2)
        )

    def forward(self, text, target_mels):
        encoder_outputs = self.encoder(text)
        mel_outputs = self.decoder(encoder_outputs, target_mels)
        
        postnet_input = mel_outputs.transpose(1, 2)
        postnet_output = self.postnet(postnet_input).transpose(1, 2)
        
        return mel_outputs + postnet_output