import torch
import os
from .charbilstm import charbilstm
from .charcnn import charcnn
from .highway import highway
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F
from .outputs import SoftmaxOutputLayer, CRFOutputLayer

import torch.nn as nn


class word_lstm(nn.Module):
    # def __init__(self, hidden_dim_word, embedding_dim_word, vocab_size_word, hidden_dim_char, embedding_dim_char,
    #              vocab_size_char, weights_word=None, is_bidirectional=True, char_level="cnn", is_highway=True,
    #              pooling='max', is_attention=False, dropout=0.1):
    def __init__(self, vocabs, embedding_dim_word, hidden_dim_word, embedding_dim_char, hidden_dim_char,
                 is_bidirectional=True, char_level="cnn", is_highway=True, use_crf=True,
                 pooling='max', is_attention=False, dropout=0.1):
        super(word_lstm, self).__init__()

        self.vocabs = vocabs
        self.weights_word = vocabs[0].vectors
        self.vocab_size_word = len(vocabs[0])

        self.vocab_size_char = len(vocabs[1])

        self.char_level = char_level
        self.charbilstm = charbilstm(hidden_dim=hidden_dim_char, embedding_dim=embedding_dim_char,
                                     vocab_size=self.vocab_size_char)
        self.charcnn = charcnn(hidden_dim=hidden_dim_char, embedding_dim=embedding_dim_char,
                               vocab_size=self.vocab_size_char)

        self.wordlstm = nn.LSTM(input_size=(embedding_dim_word + hidden_dim_char * 2), hidden_size=hidden_dim_word,
                                bidirectional=is_bidirectional,
                                batch_first=True)
        self.embeddings_word = nn.Embedding(self.vocab_size_word, embedding_dim_word)
        if self.embeddings_word is not None:
            self.embeddings_word.weight = nn.Parameter(self.weights_word, requires_grad=False)

        self.is_bidirectional = is_bidirectional
        self.pooling = pooling
        self.is_attention = is_attention
        self.is_highway = is_highway
        self.use_crf = use_crf

        if pooling == 'max' or pooling == 'mean':
            self.is_pooling = True
        else:
            self.is_pooling = False

        if is_highway:
            self.highway = highway(input_size=hidden_dim_char * 2)

        # self.linear1 = nn.Linear((is_bidirectional + self.is_attention + self.is_pooling) * 2 * hidden_dim_word, 128)
        # self.linear2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.target_size = len(vocabs[2])
        if self.use_crf:
            self.output_layer = CRFOutputLayer(
                hidden_size=((is_bidirectional+1) * hidden_dim_word),
                output_size=self.target_size)
        else:
            self.output_layer = SoftmaxOutputLayer(
                hidden_size=((is_bidirectional + 1) * hidden_dim_word),
                output_size=self.target_size)
        self.dropout = nn.Dropout(p=dropout)

    def attention_layer(self, lstm_output, final_state):
        hidden = final_state.squeeze(0)  # b x h
        # print(hidden.shape)
        attention_weight = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)  # (b x l x h) x (b x h x 1) -> b x l
        attention_weight = F.softmax(attention_weight, 1)  # b x l
        # print(attention_weight.size())
        # print(attention_weight)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), attention_weight.unsqueeze(2)).squeeze(
            2)  # (b x h x l) x (b x l x 1) -> (b x h)
        return new_hidden_state

    def loss(self, batch, compute_predictions=False):
        # x_char, x_word, seq_lengths, labels
        x_char = batch.input_char
        x_word = batch.input_word[0]
        seq_lengths = batch.input_word[1]
        labels = batch.label

        target = labels
        target = torch.autograd.Variable(target).long()

        hidden = self.compute_forward(x_char, x_word, seq_lengths)
        predictions = None
        if compute_predictions:
            predictions = self.output_layer(hidden)
        loss = self.output_layer.loss(hidden, target)
        return loss, predictions

    def save(self, path_save_model, name_model):
        checkpoint_path = os.path.join(path_save_model, name_model)
        torch.save(self.state_dict(), checkpoint_path)

    def compute_forward(self, x_char, x_word, seq_lengths):
        x_word = self.embeddings_word(x_word)
        char_feature = self.charbilstm(x_char)
        if self.char_level == 'bilstm':
            char_feature = self.charbilstm(x_char)
        elif self.char_level == 'cnn':
            char_feature = self.charcnn(x_char)

        if self.is_highway:
            char_feature = self.highway(char_feature)

        word_represent = torch.cat((x_word, char_feature), 2)
        packed_input = pack_padded_sequence(word_represent, seq_lengths, batch_first=True)

        packed_output, (ht, ct) = self.wordlstm(packed_input)

        lstm_out, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        # print(lstm_out.shape)

        # if self.is_bidirectional:
        #     ht = torch.cat((ht[-1], ht[-2]), 1)
        # else:
        #     ht = ht[-1]
        # ht_out = ht
        # if self.pooling == 'max':
        #     max_pool, _ = torch.max(lstm_out, 1)
        #     pooling_state = self.relu(max_pool)
        #     ht_out = torch.cat((ht, pooling_state), dim=1)
        # elif self.pooling == 'mean':
        #     avg_pool = torch.mean(lstm_out, 1)
        #     pooling_state = self.relu(avg_pool)
        #     ht_out = torch.cat((ht, pooling_state), dim=1)
        #
        # if self.is_attention:
        #     attention_state = self.attention_layer(lstm_out, ht)
        #     ht_out = torch.cat((ht_out, attention_state), dim=1)
        # ht_out = self.dropout(lstm_out)
        # out = self.linear1(ht_out)
        # out = self.linear2
        return lstm_out

    def forward(self, batch):
        x_char = batch.input_char
        x_word = batch.input_word[0]
        seq_lengths = batch.input_word[1]

        hidden = self.compute_forward(x_char, x_word, seq_lengths)
        predictions = self.output_layer(hidden)
        return predictions
