from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe


def load_imdb_data():
    print("Loading IMDB dataset...")
    sentence_field = data.Field(lower=True, include_lengths=True, batch_first=True, fix_length=200)
    label_field = data.Field(sequential=False)

    train, test = datasets.IMDB.splits(sentence_field, label_field)
    sentence_field.build_vocab(train, vectors=GloVe(name='6B', dim=300))
    label_field.build_vocab(train)
    # print vocab information
    print('len(TEXT.vocab)', len(sentence_field.vocab))
    print('TEXT.vocab.vectors.size()', sentence_field.vocab.vectors.size())
    print('labels ', len(label_field.vocab))
    print('Labels: ', label_field.vocab.itos)  # index to string

    train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=32, device='cpu', repeat=False,
                                                       shuffle=True)

    return train_iter, test_iter


train, test = load_imdb_data()

print(type(train))

for epoch, batch in enumerate(train):
    #print(batch)
    import model
    n = model.Neural()
    n.forward(batch.text)
    break

#batch = next(iter(train))
#print(type(batch.text[0]))
#text = batch.text[0]
#print('batch size:', text.size())
