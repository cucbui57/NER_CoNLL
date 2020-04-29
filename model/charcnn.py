import torch
import torch.nn as nn
import numpy as np


class charcnn(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, vocab_size, weights=None):
        super(charcnn, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        if weights is not None:
            self.embeddings.weight = nn.Parameter(weights, requires_grad=False)
        else:
            self.embeddings.weight = torch.from_numpy(self.random_embedding(vocab_size, hidden_dim))

        self.char_cnn = nn.Conv1d(embedding_dim, self.hidden_dim, kernel_size=3, padding=1)


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
        x = self.embeddings(x)
        char_cnn = self.char_cnn(x)
        char_out = nn.MaxPool1d(char_cnn, char_cnn.size(2)).view(batch_size, -1)
        return char_out
