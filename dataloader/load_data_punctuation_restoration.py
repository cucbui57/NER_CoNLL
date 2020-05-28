import torch
import torchtext
from torchtext import data


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


def load_data(data_path):
    input_word = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True, lower=True, include_lengths=True)

    input_char_nesting = data.Field(tokenize=list, init_token="<bos>", eos_token="<eos>",
                                    batch_first=True)

    input_char = data.NestedField(input_char_nesting,
                                  init_token="<bos>", eos_token="<eos>")

    label = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True)

    fields = [(('input_word', 'input_char'), (input_word, input_char)), ('label', label)]

    dataset = read_data(data_path, fields)
    for item in dataset:
        print(item.input_word)
        print(item.input_char)
        print(item.label)
        break


if __name__ == '__main__':
    load_data("../dataset/xxx.txt")
