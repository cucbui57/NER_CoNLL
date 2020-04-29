import torchtext
from torchtext import data, datasets
from torchtext.vocab import FastText
from torchtext.vocab import GloVe
from torchtext.vocab import CharNGram


def load_data():
    input_word = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True, lower=True)

    input_char_nesting = data.Field(tokenize=list, init_token="<bos>", eos_token="<eos>",
                                    batch_first=True)

    input_char = data.NestedField(input_char_nesting,
                                  init_token="<bos>", eos_token="<eos>")

    label = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True)

    fields = [(('input_word', 'input_char'), (input_word, input_char)),
              (None, None), ('label', label)]

    train, valid, test = datasets.CoNLL2000Chunking.splits(fields)
    input_word.build_vocab(train.input_word, vectors=GloVe(name='6B', dim=300))
    input_char.build_vocab(train.input_char)
    label.build_vocab(train.label)

    vocab_word = input_word.vocab
    word_embeddings = vocab_word.vectors
    vocab_word_size = len(vocab_word)

    vocab_char = input_char.vocab
    vocab_char_size = len(vocab_char)

    train_iter, test_iter, valid_iter = data.BucketIterator.splits((train, test, valid), batch_size=32,
                                                                   sort_key=lambda x: len(x.input_word), repeat=False,
                                                                   sort_within_batch=True, shuffle=True)
    # print(vocab_char_size)
    # print(vocab_word_size)
    # for idx, batch in enumerate(train_iter):
    #     print(batch)
    #     print(batch.input_char.size())
    #     print(batch.input_word.size())
    #     print(batch.input_word[0])
    #     print(batch.input_word[1])
    #     print(batch.input_char[0])
    #     print(batch.input_char[1])
    #     # print(batch.label)
    #     break
    return input_char, input_word, vocab_word, vocab_char, vocab_word_size, vocab_char_size, word_embeddings, train_iter, \
           valid_iter, test_iter


if __name__ == '__main__':
    load_data()
