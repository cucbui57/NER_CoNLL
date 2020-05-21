import os
import time
from load_data.dataloader import *
import torch
import torch.nn as nn

import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from model.evaluation import Evaluator, BasicMetrics, IOBMetrics
import os
from collections import defaultdict


import logging

logger = logging.getLogger(__name__)

# TEXT, vocab_size, word_embeddings, train_iter, test_iter = load_data_word_level()
CHAR, WORD, vocab_word, vocab_char, vocab_word_size, vocab_char_size, word_embeddings, char_embeddings, train_iter, test_iter = load_data_word_char_level()


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


def train_model(model, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        word = batch.word[0]
        char = batch.char[0]
        # print(char)
        # print(word)
        target = batch.label
        target = torch.autograd.Variable(target).long()

        if word.size()[0] is not 32:
            continue
        optim.zero_grad()
        prediction = model(word, char)

        loss = loss_fn(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects / len(batch)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1

        if steps % 100 == 0:
            print(
                f'Epoch: {epoch + 1}, Idx: {idx + 1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')

        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()

    return total_epoch_loss / len(train_iter), total_epoch_acc / len(train_iter)


def eval_model(model, val_iter):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            word = batch.word[0]
            char = batch.char[0]
            if word.size()[0] is not 32:
                continue
            target = batch.label
            target = torch.autograd.Variable(target).long()
            prediction = model(word, char)
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects / len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss / len(val_iter), total_epoch_acc / len(val_iter)


if __name__ == '__main__':
    learning_rate = 2e-5
    batch_size = 32
    output_size = 2
    hidden_size = 128
    embedding_length = 300
    embedding_length_char = 100

    # model = LSTM_Pooling(vocab_size=vocab_size, embedding_dim=embedding_length, hidden_dim=hidden_size,
    #                    output_size=output_size, weights=word_embeddings)

    model = LSTM_Word_Char_level(vocab_word_size, embedding_length, hidden_size, output_size, vocab_char_size,
                                 embedding_length_char, char_embeddings, word_embeddings)

    loss_fn = F.cross_entropy

    epochs = 10
    for epoch in range(epochs):
        train_loss, train_acc = train_model(model, train_iter, epoch)
        print(f'Epoch: {epoch + 1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%')

    test_loss, test_acc = eval_model(model, test_iter)
    print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')
