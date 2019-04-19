from torchtext import data
import torch
from torchtext import datasets
import torch.optim as optim
import torch.nn.functional as F
import model
import time
from log import logger


max_sequence_length = 200
device = 'cuda'
logger.info("Loading IMDB dataset...")
sentence_field = data.Field(lower=True, include_lengths=True, batch_first=True, fix_length=max_sequence_length)
label_field = data.Field(sequential=False)

train, test = datasets.IMDB.splits(sentence_field, label_field)
sentence_field.build_vocab(train)
label_field.build_vocab(train)
# print vocab information
logger.info('len(TEXT.vocab)', len(sentence_field.vocab))
# print('TEXT.vocab.vectors.size()', sentence_field.vocab.vectors.size())
logger.info('labels ', len(label_field.vocab))
logger.info('Labels: ', label_field.vocab.itos)  # index to string


batch_size = 128

train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=batch_size, device=device, repeat=False,
                                                   shuffle=False)

vocab_size = len(sentence_field.vocab)

model = model.TransformerEncoder(vocab_size, max_sequence_length)

if device == 'cuda':
    logger.info('using cuda')
    model.cuda()

model.train()
learnable_params = filter(lambda param: param.requires_grad, model.parameters())
optimizer = optim.Adam(learnable_params)
optimizer.zero_grad()
loss_function = F.cross_entropy

epoch = 20

logger.info('current memory allocated: {}'.format(torch.cuda.memory_allocated() / 1024 ** 2))
logger.info('max memory allocated: {}'.format(torch.cuda.max_memory_allocated() / 1024 ** 2))
logger.info('cached memory: {}'.format(torch.cuda.memory_cached() / 1024 ** 2))

for i in range(epoch):
    for epoch, batch in enumerate(train_iter):
        start = time.time()
        input_tensor = batch.text[0]
        predicted = model(input_tensor)
        # print('predicted:', predicted)
        #print("----------")
        # print('label', batch.label)
        label = batch.label
        loss = loss_function(predicted, label)

        loss.backward()

        optimizer.step()
        if epoch % 100 == 0:
            if torch.cuda.is_available():
                logger.info("%d iteration %d epoch with loss : %.5f in %.4f seconds" % (
                    i, epoch, loss.cpu().item(), time.time() - start))
            else:
                logger.info(("%d iteration %d epoch with loss : %.5f in %.4f seconds" % (
                    i, epoch, loss.data.numpy()[0], time.time() - start)))
