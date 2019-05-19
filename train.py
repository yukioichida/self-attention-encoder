import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, RunningAverage
from torchtext import data
from torchtext import datasets

import model
from config import *
from log import logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

logger.info("Loading IMDB dataset...")
sentence_field = data.Field(lower=True, include_lengths=True, batch_first=True, fix_length=MAX_SEQUENCE_LENGTH)
label_field = data.Field(sequential=False)

train_data, test_data = datasets.IMDB.splits(sentence_field, label_field)
train_data, valid_data = train_data.split(split_ratio=0.8, random_state=random.seed(SEED))

sentence_field.build_vocab(train_data)
label_field.build_vocab(train_data)

train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data),
                                                               batch_size=BATCH_SIZE,
                                                               device=device,
                                                               repeat=False,
                                                               shuffle=True)

vocab_size = len(sentence_field.vocab)

logger.info('vocab size: {}'.format(vocab_size))

model = model.TransformerEncoder(vocab_size, MAX_SEQUENCE_LENGTH,
                                 qty_encoder_layer=QTD_ENCODER_LAYER,
                                 qty_attention_head=ATTENTION_HEADS)

model.to(device)
if torch.cuda.is_available():
    logger.info('using cuda')
    logger.info('current memory allocated: {}'.format(torch.cuda.memory_allocated() / 1024 ** 2))
    logger.info('max memory allocated: {}'.format(torch.cuda.max_memory_allocated() / 1024 ** 2))
    logger.info('cached memory: {}'.format(torch.cuda.memory_cached() / 1024 ** 2))
else:
    logger.info("using cpu")

model.train()
learnable_params = filter(lambda param: param.requires_grad, model.parameters())
optimizer = optim.Adam(learnable_params)
optimizer.zero_grad()
loss_function = F.cross_entropy


def process_function(engine, batch):
    '''
    Function that is executed for all processed batch
    '''
    model.train()
    optimizer.zero_grad()
    x, y = batch.text[0], batch.label
    y_pred = model(x)
    loss = loss_function(y_pred, y)
    loss.backward()
    clip_grad_norm(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


def eval_function(engine, batch):
    model.eval()
    with torch.no_grad():
        x, y = batch.text[0], batch.label
        y_pred = model(x)
        return y_pred, y


trainer = Engine(process_function)
train_evaluator = Engine(eval_function)
validator_evaluator = Engine(eval_function)

RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')


def thresholded_output_transform(output):
    y_pred, y = output
    y_pred = torch.round(y_pred)
    return y_pred, y


Accuracy(output_transform=thresholded_output_transform).attach(train_evaluator, 'accuracy')
Loss(loss_function).attach(train_evaluator, 'loss_train')  # binary cross entropy

Accuracy(output_transform=thresholded_output_transform).attach(validator_evaluator, 'accuracy')
Loss(loss_function).attach(validator_evaluator, 'loss_val')

pbar = ProgressBar(persist=True, bar_format="")
pbar.attach(trainer, ['loss'])


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    train_evaluator.run(train_iter)
    metrics = train_evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_loss = metrics['loss_train']
    pbar.log_message(
        "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_loss))


def log_validation_results(engine):
    validator_evaluator.run(valid_iter)
    metrics = validator_evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_val_loss = metrics['loss_val']
    pbar.log_message(
        "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_val_loss))
    pbar.n = pbar.last_print_n = 0


trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)

# TODO: model checkpoint
trainer.run(train_iter, max_epochs=MAX_EPOCH)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def evaluate(iterator):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text

            predictions = model(text).squeeze(1)

            loss = loss_function(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


test_loss, test_acc = evaluate(test_iter)

print("Test Loss: %02d | Test Acc: %02d" % (test_loss, test_acc))
