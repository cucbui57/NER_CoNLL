import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F
import numpy as np


class charbilstm(nn.Module):
    def __init__(self, hidden_dim, vocab_size, embedding_dim, weights=None, dropout=0.1):
        super(charbilstm, self).__init__()
        self.hidden_dim = hidden_dim
        self.char_bilstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True,
                                   bidirectional=True)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if weights is not None:
            self.embedding = nn.Parameter(weights, requires_grad=False)
        else:
            self.embedding.weight.data.copy_ = torch.from_numpy(self.random_embedding(vocab_size, embedding_dim))
        self.dropout = nn.Dropout(p=dropout)

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def forward(self, x):
        seq_len = x.size(1)
        batch_size = x.size(0)
        x = x.view(batch_size * seq_len, x.size(2))
        x = self.embedding(x)
        lstm_out, (ht, ct) = self.char_bilstm(x)
        ht_out = torch.cat((ht[-1], ht[-2]), dim=1)
        ht_out = ht_out.view(batch_size, seq_len, self.hidden_dim * 2)
        ht_out = self.dropout(ht_out)
        return ht_out
