from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe


def load_imdb_data():
    print("Loading IMDB dataset...")
    sentence_field = data.Field(lower=True, include_lengths=True)
    label_field = data.Field(sequential=False)

    train, test = datasets.IMDB.splits(sentence_field, label_field)
    sentence_field.build_vocab(train, vectors=GloVe(name='6B', dim=300))
    label_field.build_vocab(train)
    # print vocab information
    print('len(TEXT.vocab)', len(sentence_field.vocab))
    print('TEXT.vocab.vectors.size()', sentence_field.vocab.vectors.size())
    print('labels ', len(label_field.vocab))
    print('Labels: ', label_field.vocab.itos)  # index to string

    train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=32, device='cuda', repeat=False,
                                                       shuffle=True)

    return train_iter, test_iter


load_imdb_data()
