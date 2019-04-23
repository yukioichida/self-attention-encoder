from torchtext import data
import torch
from torchtext import datasets
import torch.optim as optim
import torch.nn.functional as F
import model
import time
from log import logger
from config import *

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

logger.info("Loading IMDB dataset...")
sentence_field = data.Field(lower=True, include_lengths=True, batch_first=True, fix_length=max_sequence_length)
label_field = data.Field(sequential=False)

train, test = datasets.IMDB.splits(sentence_field, label_field)
sentence_field.build_vocab(train)
label_field.build_vocab(train)

train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=batch_size, device=device, repeat=False,
                                                   shuffle=True)

vocab_size = len(sentence_field.vocab)

logger.info('vocab size: {}'.format(vocab_size))

model = model.TransformerEncoder(vocab_size, max_sequence_length,
                                 qty_encoder_layer=encoder_layers,
                                 qty_attention_head=attention_heads)

if torch.cuda.is_available():
    logger.info('using cuda')
    model.cuda()
    logger.info('current memory allocated: {}'.format(torch.cuda.memory_allocated() / 1024 ** 2))
    logger.info('max memory allocated: {}'.format(torch.cuda.max_memory_allocated() / 1024 ** 2))
    logger.info('cached memory: {}'.format(torch.cuda.memory_cached() / 1024 ** 2))

model.train()
learnable_params = filter(lambda param: param.requires_grad, model.parameters())
optimizer = optim.Adam(learnable_params)
optimizer.zero_grad()
loss_function = F.cross_entropy


for i in range(max_epoch):
    for epoch, batch in enumerate(train_iter):
        start = time.time()
        input_tensor = batch.text[0]
        predicted = model(input_tensor)
        loss = loss_function(predicted, batch.label)

        loss.backward()

        optimizer.step()
        if epoch % 100 == 0:
            if torch.cuda.is_available():
                logger.info("%d iteration %d epoch with loss : %.5f in %.4f seconds" % (
                    i, i, loss.cpu().item(), time.time() - start))
            else:
                logger.info(("%d iteration %d epoch with loss : %.5f in %.4f seconds" % (
                    i, epoch, loss.data.numpy()[0], time.time() - start)))
