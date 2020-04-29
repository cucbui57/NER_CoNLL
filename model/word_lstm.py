import torch
from charbilstm import charbilstm
from charcnn import charcnn

import torch.nn as nn


class word_lstm(nn.Module):
    def __init__(self, hidden_dim_word, embedding_dim_word, vocab_size_word, hidden_dim_char, embedding_dim_char,
                 vocab_size_char, weights=None, is_bidirectional=False, char_level="bilstm"):
        super(word_lstm, self).__init__()
        self.embeddings = nn.Embedding(vocab_size_word, embedding_dim_word)
        if weights is not None:
            self.embeddings.weight = nn.Parameter(weights, requires_grad=False)
        self.lstm = nn.LSTM(input_size=(embedding_dim_word+hidden_dim_char*2), hidden_size=hidden_dim_word, bidirectional=is_bidirectional)
        self.char_level = char_level
        self.charbilstm = charbilstm(hidden_dim=hidden_dim_char, embedding_dim=embedding_dim_char,
                                     vocab_size=vocab_size_char)
        self.charcnn = charcnn()

    def forward(self, x_char, x_word):
        x_word = self.embeddings(x_word)
        char_feature = self.charbilstm(x_char)
        if self.char_level == 'bilstm':
            char_feature = self.charbilstm(x_char)
        elif self.char_level == 'cnn':
            char_feature = self.charcnn(x_char)

        word_represent = torch.cat((x_word, char_feature), 2)
        lstm_out, _ = self.lstm(word_represent)
        print(lstm_out.shape)
        return lstm_out
