import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import logging
import re

from models.evaluation import Evaluator, BasicMetrics, IOBMetrics
import os
from collections import defaultdict

from models.word_lstm import word_lstm
from dataloader.load_data_big import load_data_file, load_vocabs_pre_build
import yaml


def get_all_path_file_in_folder(path, data_type="train"):
    list_path_file = []
    for item in os.listdir(path):
        if re.search(data_type, item):
            list_path_file.append(os.path.join(path, item))
    return list_path_file


class Trainer(object):
    def __init__(self, path_save_model, model, folder_data_train, evaluator, learning_rate, path_save_vocab,use_gpu=None,
                 optimizer='adam', use_iob_metrics=True,
                 learning_rate_decay=None, name_model_checkpoint=None):
        self.path_save_model = path_save_model
        self.path_save_vocab = path_save_vocab
        self.model = model
        self.folder_data_train = folder_data_train
        # self.train_iter.repeat = False
        self.evaluator = evaluator
        self.use_iob_metrics = use_iob_metrics
        self.learning_rate = learning_rate
        self.use_gpu = use_gpu

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
        list_data_train = get_all_path_file_in_folder(self.folder_data_train)

        all_metrics = defaultdict(list)
        tag_vocab = self.model.vocabs[-1]
        # print(len(tag_vocab))
        print(tag_vocab.itos)
        metrics = [BasicMetrics(output_vocab=tag_vocab)]
        if self.use_iob_metrics:
            metrics += [IOBMetrics(tag_vocab=tag_vocab)]

        if model_check_point:
            if self.evaluator is None:
                raise ValueError("must declare evaluator for checkpoint")
            prefix_cp_model_name, monitor_metric_cp, mode_cp = self.get_model_check_point_criteria(model_check_point)

        best_value_metrics = 0

        for epoch in range(num_epochs):
            print('----------Epoch ' + str(epoch) + ' -----------------')
            epoch_loss = 0
            count = 0
            total_train_metrics = {}
            for file in list_data_train:
                print("Training data in file: " + re.search("([A-Za-z_0-9]+)(.txt)", file).group())
                train_iter = load_data_file(file, save_vocab_path=self.path_save_vocab, use_gpu=self.use_gpu)
                train_evaluator = Evaluator("train", train_iter, *metrics)

                prog_iter = tqdm(train_iter)
                for batch in prog_iter:
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
                for item in train_metrics.keys():
                    if item not in total_train_metrics:
                        total_train_metrics[item] = train_metrics[item]
                    else:
                        total_train_metrics[item] += train_metrics[item]

            for key in total_train_metrics.keys():
                total_train_metrics[key] /= len(list_data_train)

            print("\ntrain metric here: ", total_train_metrics)

            if self.evaluator:
                eval_metrics = self.evaluator.evaluate(self.model)
                print("valid metric here: ", eval_metrics)
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
                    name_model = "model_epoch_{}.pt".format(epoch)
                    self.model.save(self.path_save_model, name_model)
            else:
                name_model = "model_epoch_{}.pt".format(epoch)
                self.model.save(self.path_save_model, name_model)
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
    config = load_config_file('config_train.yaml')

    path_save_model = config['path_save_model']
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    optimizer = config['optimizer']
    use_iob_metrics = config['use_iob_metrics']
    path_model_checkpoint = config['path_model_checkpoint']
    path_save_vocab = config['path_save_vocab']
    folder_data_train = config['folder_data_train']
    path_data_test = config['path_data_test']
    use_gpu = config['use_gpu']

    if use_gpu > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = use_gpu

    vocabs = load_vocabs_pre_build(path_save_vocabs=path_save_vocab)
    vocabs = vocabs['vocabs']
    tag_vocab = vocabs[-1]

    metrics = [BasicMetrics(output_vocab=tag_vocab)]
    if use_iob_metrics:
        metrics += [IOBMetrics(tag_vocab=tag_vocab)]

    valid_iter = load_data_file(path_data_test, save_vocab_path=path_save_vocab, use_gpu=use_gpu)

    evaluator = Evaluator("valid", valid_iter, *metrics)

    if path_model_checkpoint == "None":
        model = word_lstm.create(path_save_model, config, vocabs)
    else:
        model = word_lstm.load(path_save_model, path_model_checkpoint, config)

    if use_gpu >= 0:
        model = model.cuda(device=use_gpu)

    trainer = Trainer(path_save_model=path_save_model, model=model, folder_data_train=folder_data_train,
                      evaluator=evaluator, path_save_vocab=path_save_vocab, use_gpu=use_gpu,
                      optimizer=optimizer, learning_rate=learning_rate, use_iob_metrics=use_iob_metrics)
    check_point = 'bilstm_crf__valid_F1__max'
    trainer.train(num_epochs=num_epochs, model_check_point=None)
    # trainer.evaluate()
