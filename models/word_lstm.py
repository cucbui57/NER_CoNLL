import torch
import os
from .charbilstm import charbilstm
from .charcnn import charcnn
from .highway import highway
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F
from .outputs import SoftmaxOutputLayer, CRFOutputLayer
from .normalization import LayerNorm
import json

import torch.nn as nn

NAME_CONFIG = 'model_parameter.conf'
NAME_VOCABS = 'vocabs.pt'


def xavier_uniform_init(m):
    """
    Xavier initializer to be used with model.apply
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight.data)


class word_lstm(nn.Module):
    # def __init__(self, vocabs, embedding_dim_word, hidden_dim_word, embedding_dim_char, hidden_dim_char,
    #              is_bidirectional=True, char_level="cnn", is_highway=True, use_crf=True, is_attention=True,
    #              dropout=0.1, use_norm_before_attention=True, use_norm_before_hidden=True, use_residual=True,
    #              add_hidden_layer_component=True):
    def __init__(self, vocabs, config):
        super(word_lstm, self).__init__()

        self.use_char = config['use_char']
        self.vocabs = vocabs

        self.embedding_dim_word = config['embedding_dim_word']
        self.hidden_dim_word = config['hidden_dim_word']
        self.is_bidirectional = config['is_bidirectional']
        self.weights_word = vocabs[0].vectors
        self.vocab_size_word = len(vocabs[0])

        if self.use_char:
            self.wordlstm = nn.LSTM(input_size=(self.embedding_dim_word + self.hidden_dim_char * 2),
                                    hidden_size=self.hidden_dim_word,
                                    bidirectional=self.is_bidirectional,
                                    batch_first=True)
        else:
            self.wordlstm = nn.LSTM(input_size=self.embedding_dim_word,
                                    hidden_size=self.hidden_dim_word,
                                    bidirectional=self.is_bidirectional,
                                    batch_first=True)

        self.embeddings_word = nn.Embedding(self.vocab_size_word, self.embedding_dim_word)

        if self.embeddings_word is not None:
            self.embeddings_word.weight = nn.Parameter(self.weights_word, requires_grad=False)

        if self.use_char:
            self.embedding_dim_char = config['embedding_dim_char']
            self.hidden_dim_char = config['hidden_dim_char']
            self.char_level = config['char_level']
            self.is_highway = config['is_highway']
            self.vocab_size_char = len(vocabs[1])

            self.charbilstm = charbilstm(hidden_dim=self.hidden_dim_char, embedding_dim=self.embedding_dim_char,
                                         vocab_size=self.vocab_size_char)
            self.charcnn = charcnn(hidden_dim=self.hidden_dim_char, embedding_dim=self.embedding_dim_char,
                                   vocab_size=self.vocab_size_char)
            if self.is_highway:
                self.highway = highway(input_size=self.hidden_dim_char * 2)

        self.is_attention = config['is_attention']
        self.use_residual = config['use_residual']
        self.use_norm_before_hidden = config['use_norm_before_hidden']
        self.use_norm_before_attention = config['use_norm_before_attention']
        self.add_hidden_layer_component = config['add_hidden_layer_component']
        self.layer_norm = LayerNorm(self.hidden_dim_word * 2)

        self.use_crf = config['use_crf']
        self.pooling = config['pooling']
        self.dropout = nn.Dropout(p=config['drop_out'])
        self.relu = nn.ReLU()
        self.target_size = len(vocabs[-1])
        self.add_hidden_layer_component = self.add_hidden_layer_component

        if self.add_hidden_layer_component:
            self.hidden_layer_component = nn.Sequential(nn.Linear((2 * (
                    self.is_bidirectional + self.is_attention) * self.hidden_dim_word + self.use_residual * self.embedding_dim_word),
                                                                  512),
                                                        self.relu,
                                                        self.dropout,
                                                        nn.Linear(512, 256),
                                                        self.relu,
                                                        self.dropout)
            if self.use_crf:
                self.output_layer = CRFOutputLayer(
                    hidden_size=256, output_size=self.target_size)
            else:
                self.output_layer = SoftmaxOutputLayer(
                    hidden_size=256, output_size=self.target_size)
        else:
            if self.use_crf:
                self.output_layer = CRFOutputLayer(
                    hidden_size=(2 * (
                            self.is_bidirectional + self.is_attention) * self.hidden_dim_word + self.use_residual * self.embedding_dim_word),
                    output_size=self.target_size)
            else:
                self.output_layer = SoftmaxOutputLayer(
                    hidden_size=(2 * (
                            self.is_bidirectional + self.is_attention) * self.hidden_dim_word + self.use_residual * self.embedding_dim_word),
                    output_size=self.target_size)

    def loss(self, batch, compute_predictions=False):
        # x_char, x_word, seq_lengths, labels
        labels = batch.label

        target = labels
        target = torch.autograd.Variable(target).long()

        hidden = self.compute_forward(batch)
        predictions = None
        if compute_predictions:
            predictions = self.output_layer(hidden)
        loss = self.output_layer.loss(hidden, target)
        return loss, predictions

    def save(self, path_save_model, name_model):
        checkpoint_path = os.path.join(path_save_model, name_model)
        torch.save(self.state_dict(), checkpoint_path)

    def self_attention(self, lstm_output):  # batch x seq_len x (hidden_layer*num_direction)
        lstm_output_transpose = lstm_output.permute(0, 2, 1)  # batch x (hidden_layer*num_direction) x seq_len
        attn_weights = torch.matmul(lstm_output, lstm_output_transpose)
        soft_attn_weights = F.softmax(attn_weights, -1)
        hidden_state_self_attn = torch.matmul(soft_attn_weights, lstm_output)
        new_hidden_state = torch.cat([lstm_output, hidden_state_self_attn], dim=-1)
        return new_hidden_state

    def compute_forward(self, batch):

        x_word = batch.input_word[0]
        seq_lengths = batch.input_word[1]

        x_word = self.embeddings_word(x_word)

        if self.use_char:
            x_char = batch.input_char
            char_feature = self.charbilstm(x_char)
            if self.char_level == 'bilstm':
                char_feature = self.charbilstm(x_char)
            elif self.char_level == 'cnn':
                char_feature = self.charcnn(x_char)

            if self.is_highway:
                char_feature = self.highway(char_feature)

            word_represent = torch.cat((x_word, char_feature), 2)
        else:
            word_represent = x_word

        packed_input = pack_padded_sequence(word_represent, seq_lengths, batch_first=True)

        packed_output, (ht, ct) = self.wordlstm(packed_input)

        lstm_out, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        if self.use_norm_before_attention:
            lstm_out = self.layer_norm(lstm_out)

        if self.is_attention:
            lstm_out = self.self_attention(self.dropout(lstm_out))
        else:
            lstm_out = self.dropout(lstm_out)

        if self.use_residual:
            lstm_out = self.dropout(torch.cat((lstm_out, x_word), dim=-1))
        else:
            lstm_out = self.dropout(lstm_out)

        if self.add_hidden_layer_component:
            lstm_out = self.hidden_layer_component(lstm_out)
        return lstm_out

    @classmethod
    def create(cls, path_folder_model, config, vocabs):
        model = cls(vocabs, config)
        model.apply(xavier_uniform_init)
        if torch.cuda.is_available():
            model = model.cuda()

        path_vocab_file = os.path.join(path_folder_model, NAME_VOCABS)
        torch.save(vocabs, path_vocab_file)

        path_config_file = os.path.join(path_folder_model, NAME_CONFIG)
        with open(path_config_file, "w") as w_config:
            json.dump(config, w_config)

        return model

    @classmethod
    def load(cls, path_folder_model, path_model_checkpoint, config):
        path_vocab_file = os.path.join(path_folder_model, NAME_VOCABS)

        vocabs = torch.load(path_vocab_file)

        model = cls(vocabs, config)
        if torch.cuda.is_available():
            model = model.cuda()
            model.load_state_dict(torch.load(path_model_checkpoint))
        else:
            model.load_state_dict(torch.load(path_model_checkpoint, map_location=lambda storage, loc: storage))
        return model

    def forward(self, batch):
        hidden = self.compute_forward(batch)
        predictions = self.output_layer(hidden)
        return predictions
