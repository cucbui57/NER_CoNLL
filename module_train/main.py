import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from model.evaluation import Evaluator, BasicMetrics, IOBMetrics
import os
from collections import defaultdict

from model.word_lstm import word_lstm
from dataloader.load_data import load_data


class Trainer(object):
    def __init__(self, path_save_model, model, train_iter, evaluator, optimizer='adam', learning_rate_decay=None):
        self.path_save_model = path_save_model
        self.model = model
        self.train_iter = train_iter
        self.train_iter.repeat = False
        self.evaluator = evaluator

        model_params = filter(lambda p: p.requires_grad, model.parameters())

        if optimizer == "adam":
            self.optimizer = optim.Adam(
                model_params,
                betas=(0.9, 0.99),
                lr=0.01,
                weight_decay=0
            )
        else:
            self.optimizer = optim.SGD(
                model_params,
                lr=0.01,
                momentum=0.9,
                weight_decay=0
            )

    def train(self, num_epochs):
        all_metrics = defaultdict(list)
        tag_vocab = self.model.vocabs[2]

        metrics = [BasicMetrics(output_vocab=tag_vocab)]
        metrics += [IOBMetrics(tag_vocab=tag_vocab)]

        train_evaluator = Evaluator("train", self.train_iter, *metrics)

        best_value_metrics = 0

        for epoch in range(num_epochs):
            print('----------Epoch ' + str(epoch) + ' -----------------')
            self.train_iter.init_epoch()  # todo init_epoch() ???

            epoch_loss = 0
            prog_iter = tqdm(self.train_iter)
            count = 0

            for batch in prog_iter:
                # print(batch)
                # print(batch.input_word[0].size())
                # print(batch.input_word[1].size())
                self.model.train()
                self.optimizer.zero_grad()
                loss, _ = self.model.loss(batch)
                loss.backward()

                nn.utils.clip_grad_norm(self.model.parameters(), 5.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                count += 1
                prog_iter.set_description('Training')
                prog_iter.set_postfix(loss=(epoch_loss / count))

            train_loss = epoch_loss / count
            all_metrics['train_loss'].append(train_loss)
            train_metrics = train_evaluator.evaluate(self.model)
            print("\ntrain metric here: ", train_metrics)

            if self.evaluator:
                eval_metrics = self.evaluator.evaluate(self.model)
                print("test metric here: ", eval_metrics)
                if not isinstance(eval_metrics, dict):
                    raise ValueError('eval_fn should return a dict of metrics')

            else:
                print("epoch {} train F1: {}, precision: {},"
                      " recall: {} *** ".format(epoch,
                                                round(train_metrics['F1'], 3),
                                                round(train_metrics['precision'], 3),
                                                round(train_metrics['recall'], 3)))
            self.model.save(self.path_save_model, 'checkpoint_' + str(epoch))

    def evaluate(self):
        eval_metrics = self.evaluator.evaluate(self.model)
        print('test metric here: ', eval_metrics)


if __name__ == '__main__':
    dataset = load_data()
    # print(dataset)
    use_iob_metrics = True

    _, _, tag_vocab = dataset['vocabs']
    metrics = [BasicMetrics(output_vocab=tag_vocab)]
    if use_iob_metrics:
        metrics += [IOBMetrics(tag_vocab=tag_vocab)]

    train_iter = dataset["iter"][0]
    valid_iter = dataset["iter"][1]
    test_iter = dataset["iter"][2]

    # evaluator = Evaluator("validation", valid_iter, *metrics)
    evaluator = Evaluator("testing", test_iter, *metrics)
    path_save_model = "save_models/"
    vocabs = dataset['vocabs']
    embedding_dim_word = 300
    embedding_dim_char = 150
    hidden_dim_word = 100
    hidden_dim_char = 50
    num_epochs = 1

    model = word_lstm(vocabs, embedding_dim_word, hidden_dim_word, embedding_dim_char, hidden_dim_char, is_highway=True,
                      is_bidirectional=True, char_level="cnn", use_crf=True, pooling='no', is_attention=False,
                      dropout=0.1)
    model.load_state_dict(torch.load("save_models/checkpoint_0"))

    trainer = Trainer(path_save_model, model, train_iter, evaluator, optimizer='adam')

    trainer.train(num_epochs=num_epochs)
    trainer.evaluate()
