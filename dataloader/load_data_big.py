from torchtext import data
from torchtext.vocab import Vectors
import torchtext
import logging
import os
import torch
import re

from collections import defaultdict

logger = logging.getLogger(__name__)

COMMA_SIGN = 'C'
NO_SIGN = 'O'
PERIOD_SIGN = 'P'


class MyPretrainedVector(Vectors):
    def __init__(self, name_file, cache):
        super(MyPretrainedVector, self).__init__(name_file, cache=cache)


def read_data(corpus_file, datafields):
    with open(corpus_file, encoding='utf-8') as f:
        examples = []
        words = []
        labels = []
        for line in f:
            line = line.strip()
            if not line:
                examples.append(torchtext.data.Example.fromlist([words, labels], datafields))
                words = []
                labels = []
            else:
                columns = line.split()
                words.append(columns[0])
                labels.append(columns[-1])
        return torchtext.data.Dataset(examples, datafields)


def get_all_path_file_in_folder(path):
    list_path_file = []
    for item in os.listdir(path):
        list_path_file.append(os.path.join(path, item))
    return list_path_file


def make_vocab_each_file(path_folder_dataset, path_folder_save_vocab, min_freq=10):
    list_path_file = get_all_path_file_in_folder(path_folder_dataset)

    for e_path_file in list_path_file:
        name_file = re.search("([a-z_0-9]+)(.txt)", e_path_file)
        new_name_file = name_file.group().replace(".txt", "_vocab.pt")
        # print(new_name_file)
        new_path_file = os.path.join(path_folder_save_vocab, new_name_file)

        input_word = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True)
        label = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True)
        fields = [('input_word', input_word), ('label', label)]

        dataset = read_data(e_path_file, fields)

        # Build vocab
        input_word.build_vocab(dataset.input_word, min_freq=min_freq)

        print(input_word.vocab.freqs)

        logger.warning('Input vocab size:%d' % (len(input_word.vocab)))
        logger.warning("Load dataset done")
        torch.save(input_word.vocab, new_path_file)


def combine_vocab_all_file(path_folder_vocab, path_save_vocab, name_vocab=None, cache_folder=None, min_freqs=10):
    '''

    :param path_folder_vocab: path of folder vocab
    :param path_save_vocab: path save vocab
    :param name_vocab:  file name pretrained embedding
    :param cache_folder :use for load pretrained model embedding (folder save file pre)
    :param min_freqs: min freq word need appear for can be in dictionary
    :return:
    '''
    list_path_file = get_all_path_file_in_folder(path_folder_vocab)

    dict_combine_vocab = defaultdict(lambda: 0)

    for e_path_file in list_path_file:
        e_vocab = torch.load(e_path_file)
        for e_key, e_value in dict(e_vocab.freqs).items():
            if e_key not in dict_combine_vocab:
                dict_combine_vocab[e_key] = e_value
            else:
                dict_combine_vocab[e_key] += e_value
    # print(dict_combine_vocab)

    # build vocab for input, it has token [cls] [sep] and [mask]
    with open("tmp_vocab_input.txt", "w") as wf:
        for e_key, e_value in dict_combine_vocab.items():
            if e_value >= min_freqs:
                wf.write(e_key + "\n")

    input_word = data.Field(batch_first=True)
    fields = [('input_word', input_word)]
    dataset = data.TabularDataset("tmp_vocab_input.txt",
                                  format='csv',
                                  fields=fields,
                                  csv_reader_params={'delimiter': '\n',
                                                     'quoting': 3})
    input_word.build_vocab(dataset, vectors=[MyPretrainedVector(name_vocab, cache_folder)])

    # build vocab for label

    with open("tmp_label.txt", "w") as wf:
        wf.write("{} {} {}".format(COMMA_SIGN, PERIOD_SIGN, NO_SIGN))

    label = data.Field(batch_first=True)
    fields = [('label', label)]
    dataset_label = data.TabularDataset(
        path="tmp_label.txt",
        format='csv',
        fields=fields,
        csv_reader_params={'delimiter': '|',
                           'quoting': 3}
    )
    label.build_vocab(dataset_label, specials_first=False)
    print(label.vocab.stoi)

    os.remove("tmp_vocab_input.txt")
    os.remove("tmp_label.txt")

    # save vocab input label
    path_vocab_input = os.path.join(path_save_vocab, "vocab_input.pt")
    path_vocab_label = os.path.join(path_save_vocab, "vocab_label.pt")

    torch.save(input_word.vocab, path_vocab_input)
    torch.save(label.vocab, path_vocab_label)


def load_vocabs_pre_build(path_save_vocabs):
    vocabs_input = torch.load(path_save_vocabs + "/vocab_input.pt")
    label = torch.load(path_save_vocabs + "/vocab_label.pt")
    vocabs = {
        "vocabs": (vocabs_input, label)
    }
    return vocabs


def load_data_file(data_path, save_vocab_path):
    input_word = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True, include_lengths=True)
    label = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True)

    fields = [('input_word', input_word), ('label', label)]

    dataset = read_data(data_path, fields)

    vocabs = load_vocabs_pre_build(path_save_vocabs=save_vocab_path)
    vocabs = vocabs['vocabs']
    input_word.vocab = vocabs[0]
    label.vocab = vocabs[-1]

    dataset_iter = data.BucketIterator(dataset=dataset,
                                       batch_size=32,
                                       sort_key=lambda x: len(x.input_word), repeat=False,
                                       sort_within_batch=True, shuffle=True,
                                       device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                                       )
    # print(dataset_iter)
    # for batch in dataset_iter:
    #     print(batch.input_word[0])
    #     break
    return dataset_iter


if __name__ == '__main__':
    # list_path = get_all_path_file_in_folder("../dataset/data_train_split")
    # print(list_path)
    make_vocab_each_file("../dataset/data_train_split", "../dataset/save_vocab/train_vocab/")
    combine_vocab_all_file("../dataset/save_vocab/train_vocab", "../dataset/save_vocab", name_vocab="embedding.txt",
                           cache_folder="../dataset/save_vocab/cache/")

    # load_data_file("../dataset/data_train_split/train_00.txt", "../dataset/save_vocab")
