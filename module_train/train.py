import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from models.evaluation import Evaluator, BasicMetrics, IOBMetrics
import os
from collections import defaultdict

from models.word_char_lstm import word_lstm
from dataloader.load_data import load_data
import yaml


class Trainer(object):
    def __init__(self, path_save_model, model, train_iter, evaluator, learning_rate, optimizer='adam',
                 learning_rate_decay=None, name_model_checkpoint=None):
        self.path_save_model = path_save_model
        self.model = model
        self.train_iter = train_iter
        self.train_iter.repeat = False
        self.evaluator = evaluator
        self.learning_rate = learning_rate

        model_params = filter(lambda p: p.requires_grad, model.parameters())

        if optimizer == "adam":
            self.optimizer = optim.Adam(
                model_params,
                betas=(0.9, 0.99),
                lr=self.learning_rate,
                weight_decay=0
            )
        else:
            self.optimizer = optim.SGD(
                model_params,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=0
            )

    @staticmethod
    def get_model_check_point_criteria(model_check_point):
        check_point = model_check_point.split("__")
        if len(check_point) != 3:
            raise ValueError("malformed checkpoint criteria")

        prefix_model_name, monitor_metric, mode = check_point
        return prefix_model_name, monitor_metric, mode

    def train(self, num_epochs, model_check_point=None):
        all_metrics = defaultdict(list)
        tag_vocab = self.model.vocabs[2]
        # print(len(tag_vocab))
        print(tag_vocab.itos[0:25])
        metrics = [BasicMetrics(output_vocab=tag_vocab)]
        metrics += [IOBMetrics(tag_vocab=tag_vocab)]

        train_evaluator = Evaluator("train", self.train_iter, *metrics)

        if model_check_point:
            if self.evaluator is None:
                raise ValueError("must declare evaluator for checkpoint")
            prefix_cp_model_name, monitor_metric_cp, mode_cp = self.get_model_check_point_criteria(model_check_point)

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

                if model_check_point:
                    if "train" in monitor_metric_cp:
                        name_metric = monitor_metric_cp.replace("train_", "")
                        spec_dataset_metric = round(train_metrics[name_metric], 3)
                        other_dataset_metric = round(eval_metrics[name_metric], 3)
                        name_model = "{}_epoch_{}_{}_train_{}_val_{}".format(prefix_cp_model_name,
                                                                             count,
                                                                             name_metric,
                                                                             spec_dataset_metric,
                                                                             other_dataset_metric)

                    elif "val" in monitor_metric_cp:
                        name_metric = monitor_metric_cp.replace("valid_", "")
                        spec_dataset_metric = round(eval_metrics[name_metric], 3)
                        other_dataset_metric = round(train_metrics[name_metric], 3)
                        name_model = "{}_epoch_{}_{}_train_{}_val_{}".format(prefix_cp_model_name,
                                                                             count,
                                                                             name_metric,
                                                                             other_dataset_metric,
                                                                             spec_dataset_metric)
                    else:
                        raise ValueError("key should be train or val in model checkpoint")

                    if mode_cp == "min":
                        if spec_dataset_metric < best_value_metrics:
                            best_value_metrics = spec_dataset_metric
                            self.model.save(self.path_save_model, name_model)

                            arr_name_model = name_model.split("_")
                            name_optimizer = "optimizer_" + "_".join(arr_name_model[1:])
                            path_save_optimizer = os.path.join(self.path_save_model, name_optimizer)
                            torch.save(self.optimizer.state_dict(), path_save_optimizer)

                    if mode_cp == "max":
                        if spec_dataset_metric > best_value_metrics:
                            best_value_metrics = spec_dataset_metric
                            self.model.save(self.path_save_model, name_model)

                            arr_name_model = name_model.split("_")
                            name_optimizer = "optimizer_" + "_".join(arr_name_model[1:])
                            path_save_optimizer = os.path.join(self.path_save_model, name_optimizer)
                            torch.save(self.optimizer.state_dict(), path_save_optimizer)
            else:
                print("epoch {} train F1: {}, precision: {},"
                      " recall: {} *** ".format(epoch,
                                                round(train_metrics['F1'], 3),
                                                round(train_metrics['precision'], 3),
                                                round(train_metrics['recall'], 3)))

    def evaluate(self):
        eval_metrics = self.evaluator.evaluate(self.model)
        print('test metric here: ', eval_metrics)


def load_config_file(configuration_directory):
    with open(configuration_directory, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

        return config


if __name__ == '__main__':
    dataset = load_data()
    config = load_config_file('config_train.yaml')

    path_save_model = config['path_save_model']

    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    optimizer = config['optimizer']
    use_iob_metrics = config['use_iob_metrics']

    path_model_checkpoint = config['path_model_checkpoint']

    _, _, tag_vocab = dataset['vocabs']
    metrics = [BasicMetrics(output_vocab=tag_vocab)]
    if use_iob_metrics:
        metrics += [IOBMetrics(tag_vocab=tag_vocab)]

    train_iter = dataset["iter"][0]
    valid_iter = dataset["iter"][1]
    test_iter = dataset["iter"][2]

    # evaluator = Evaluator("validation", valid_iter, *metrics)
    evaluator = Evaluator("testing", test_iter, *metrics)
    vocabs = dataset['vocabs']

    # model = word_lstm(vocabs, config)
    print(type(path_model_checkpoint))
    if path_model_checkpoint == "None":
        model = word_lstm.create(path_save_model, config, vocabs)
    else:
        model = word_lstm.load(path_save_model, path_model_checkpoint, config)

    trainer = Trainer(path_save_model=path_save_model, model=model, train_iter=train_iter, evaluator=evaluator,
                      optimizer=optimizer, learning_rate=learning_rate)
    check_point = 'blstm_atten2016_origin_single__valid_F1__max'
    trainer.train(num_epochs=num_epochs, model_check_point=None)
    # trainer.evaluate()
