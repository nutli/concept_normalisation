#! /usr/bin/env python
__author__ = 'nl347'

import numpy as np
from tensorflow.contrib import learn
import lasagne
from gensim.models.word2vec import Word2Vec
import theano
import logging
import sys
import time
import argparse

logger = logging.getLogger("concept_normalisation")
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)

def load_data(training_file, validation_file, testing_file):
    # Load data from files
    texts = []
    labels = []
    for file_n in [training_file, validation_file, testing_file]:
        txts = []
        lbs = []
        with open(file_n,'r') as f:
            for line in f:
                line = line.strip()
                label, _, text = line.split("\t")
                txts.append(text)
                lbs.append(label)
        texts.append(txts)
        labels.append(lbs)

    return texts[0], texts[1], texts[2], labels[0], labels[1], labels[2]

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def load_word_embedding_dict(embedding, embedding_path, words_dict, logger, embedd_dim=100):
    # loading word2vec
    if embedding == 'word2vec':
        logger.info("Loading word2vec ...")
        word2vec = Word2Vec.load_word2vec_format(embedding_path, binary=True, unicode_errors='ignore')
        embedd_dim = word2vec.vector_size
        return word2vec, embedd_dim
    elif embedding == 'random':
        # loading random embedding table
        logger.info("Loading Random ...")
        embedd_dict = dict()
        words = words_dict
        scale = np.sqrt(3.0 / embedd_dim)
        for word in words:
            embedd_dict[word] = np.random.uniform(-scale, scale, [1, embedd_dim])
        return embedd_dict, embedd_dim

def build_embedd_table(word_dict, embedd_dict, embedd_dim):
    scale = np.sqrt(3.0 / embedd_dim)
    embedd_table = np.empty([len(word_dict), embedd_dim], dtype=theano.config.floatX)
    embedd_table[0, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
    index = 0
    for word in word_dict:
        ww = word
        embedd = embedd_dict[ww] if ww in embedd_dict else np.random.uniform(-scale, scale, [1, embedd_dim])
        embedd_table[index, :] = embedd
        index = index +1
    return embedd_table

def categorical_accuracy(predictions, targets, top_k=1):
    """Computes the categorical accuracy between predictions and targets.
    .. math:: L_i = \\mathbb{I}(t_i = \\operatorname{argmax}_c p_{i,c})
    Can be relaxed to allow matches among the top :math:`k` predictions:
    .. math::
        L_i = \\mathbb{I}(t_i \\in \\operatorname{argsort}_c (-p_{i,c})_{:k})
    Parameters
    ----------
    predictions : Theano 2D tensor
        Predictions in (0, 1), such as softmax output of a neural network,
        with data points in rows and class probabilities in columns.
    targets : Theano 2D tensor or 1D tensor
        Either a vector of int giving the correct class index per data point
        or a 2D tensor of 1 hot encoding of the correct class in the same
        layout as predictions
    top_k : int
        Regard a prediction to be correct if the target class is among the
        `top_k` largest class probabilities. For the default value of 1, a
        prediction is correct only if the target class is the most probable.
    Returns
    -------
    Theano 1D tensor
        An expression for the item-wise categorical accuracy in {0, 1}
    Notes
    -----
    This is a strictly non differential function as it includes an argmax.
    This objective function should never be used with a gradient calculation.
    It is intended as a convenience for validation and testing not training.
    To obtain the average accuracy, call :func:`theano.tensor.mean()` on the
    result, passing ``dtype=theano.config.floatX`` to compute the mean on GPU.
    """
    if targets.ndim == predictions.ndim:
        targets = theano.tensor.argmax(targets, axis=-1)
    elif targets.ndim != predictions.ndim - 1:
        raise TypeError('rank mismatch between targets and predictions')

    if top_k == 1:
        # standard categorical accuracy
        top = theano.tensor.argmax(predictions, axis=-1)
        return theano.tensor.eq(top, targets)
    else:
        # top-k accuracy
        top = theano.tensor.argsort(predictions, axis=-1)
        # (Theano cannot index with [..., -top_k:], we need to simulate that)
        top = top[[slice(None) for _ in range(top.ndim - 1)] +
                  [slice(-top_k, None)]]
        targets = theano.tensor.shape_padaxis(targets, axis=-1)
        return theano.tensor.any(theano.tensor.eq(top, targets), axis=-1)


parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dict', default='data/word2vec/GoogleNews-vectors-negative300.bin')
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--train', default='AskAPatient.fold-0.train.txt')
parser.add_argument('--dev', default='AskAPatient.fold-0.validation.txt')
parser.add_argument('--test', default='AskAPatient.fold-0.test.txt')
parser.add_argument('--embedding', default='random')
parser.add_argument('--network', default='rnn', help='Support cnn, rnn and ff')
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--gamma', type=float, default=1e-6)
parser.add_argument('--num_epochs', type=int, default=50)

args = parser.parse_args()
train_path = args.train
dev_path = args.dev
test_path = args.test
embedding = args.embedding
network = args.network
learning_rate = args.learning_rate
gamma = args.gamma
batch_size = args.batch_size
num_epochs = args.num_epochs

# Load data
logger.info("Loading data...")
x_train_text, x_validation_text, x_test_text, y_train_text, y_validation_text, y_test_text = load_data(train_path, dev_path, test_path)

# Build vocabulary
max_length = 56 #max_length of sentences
vocab_processor = learn.preprocessing.VocabularyProcessor(max_length)
all_text = x_train_text+x_validation_text+x_test_text

X = np.array(list(vocab_processor.fit_transform(all_text)), dtype=np.int32)
X_train = X[0:len(x_train_text)]
X_validation = X[len(x_train_text):len(x_train_text)+len(x_validation_text)]
X_test = X[len(x_train_text)+len(x_validation_text):]

categorical_processor = learn.preprocessing.VocabularyProcessor(1)
all_label = y_train_text+y_validation_text+y_test_text
num_labels = len(set(all_label))

y = np.array(list(categorical_processor.fit_transform(all_label)), dtype=np.int32)
y_train = y[0:len(y_train_text)]
y_validation = y[len(y_train_text):len(y_train_text)+len(y_validation_text)]
y_test = y[len(y_train_text)+len(y_validation_text):]

y_train_t = list(y_train.reshape(len(y_train)))
y_validation_t = list(y_validation.reshape(len(y_validation)))
y_test_t = list(y_test.reshape(len(y_test)))

y_train = np.zeros((len(y_train), num_labels), dtype=np.int32)
y_train[np.arange(len(y_train)), [v-1 for v in y_train_t]] = 1

y_validation = np.zeros((len(y_validation), num_labels), dtype=np.int32)
y_validation[np.arange(len(y_validation)), [v-1 for v in y_validation_t]] = 1

y_test = np.zeros((len(y_test), num_labels), dtype=np.int32)
y_test[np.arange(len(y_test)), [v-1 for v in y_test_t]] = 1

words_dict = [vocab_processor.vocabulary_.reverse(i) for i in xrange(len(vocab_processor.vocabulary_))]
embedd_dict, embedd_dim = load_word_embedding_dict("random", '/path', words_dict, logger, embedd_dim=300)
embedd_dict["<UNK>"] = np.zeros(embedd_dim)

embedd_table = build_embedd_table(words_dict, embedd_dict, embedd_dim)

logger.info("constructing network...")

input_var = theano.tensor.imatrix(name='inputs')
target_var_flatten = theano.tensor.imatrix(name='targets')
num_data, max_length = X_train.shape

if network == 'rnn':
    layer_input = lasagne.layers.InputLayer(shape=(None, max_length), input_var=input_var, name='input')
    layer_embedding = lasagne.layers.EmbeddingLayer(layer_input, input_size=len(words_dict),
                                                    output_size=embedd_dim,
                                                    W=embedd_table, name='embedding')
    layer_GRU = lasagne.layers.GRULayer(layer_embedding, num_units=100)

    layer_hidden = lasagne.layers.DenseLayer(lasagne.layers.dropout(layer_GRU, p=.5), num_units=100,
                                             nonlinearity=lasagne.nonlinearities.rectify)

    layer_output = lasagne.layers.DenseLayer(
                layer_hidden,
                num_units=num_labels,
                nonlinearity=lasagne.nonlinearities.softmax)
elif network == 'cnn':
    layer_input = lasagne.layers.InputLayer(shape=(None, max_length), input_var=input_var, name='input')
    layer_embedding = lasagne.layers.EmbeddingLayer(layer_input, input_size=len(words_dict),
                                                    output_size=embedd_dim,
                                                    W=embedd_table, name='embedding')
    layer_convs = []
    for filter_h in [3, 4, 5]:
        layer_conv = lasagne.layers.Conv1DLayer(
                    layer_embedding, num_filters=100, filter_size=filter_h,
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.GlorotUniform())

        _, _, pool_size = layer_conv.output_shape
        layer_conv = lasagne.layers.MaxPool1DLayer(layer_conv, pool_size=pool_size)
        layer_convs.append(layer_conv)

    layer_conv = lasagne.layers.concat(layer_convs)

    print 'layer_conv output', layer_conv.output_shape

    layer_hidden = lasagne.layers.DenseLayer(lasagne.layers.dropout(layer_conv, p=.5), num_units=100,
                                             nonlinearity=lasagne.nonlinearities.rectify)

    layer_output = lasagne.layers.DenseLayer(
                layer_hidden,
                num_units=num_labels,
                nonlinearity=lasagne.nonlinearities.softmax)
else: #default is ff
    layer_input = lasagne.layers.InputLayer(shape=(None, max_length), input_var=input_var, name='input')
    layer_embedding = lasagne.layers.EmbeddingLayer(layer_input, input_size=len(words_dict),
                                                    output_size=embedd_dim,
                                                    W=embedd_table, name='embedding')

    layer_output = lasagne.layers.DenseLayer(
                layer_embedding,
                num_units=num_labels,
                nonlinearity=lasagne.nonlinearities.softmax)

prediction_train = lasagne.layers.get_output(layer_output)
prediction_eval = lasagne.layers.get_output(layer_output, deterministic=True)
final_prediction = theano.tensor.argmax(prediction_eval, axis=1)

loss_train = lasagne.objectives.categorical_crossentropy(prediction_train, target_var_flatten).sum(dtype=theano.config.floatX)
l2_penalty = lasagne.regularization.regularize_network_params(layer_output, lasagne.regularization.l2)
loss_train = loss_train + gamma * l2_penalty

loss_eval = lasagne.objectives.categorical_crossentropy(prediction_eval, target_var_flatten).sum(dtype=theano.config.floatX)
corr_train = categorical_accuracy(prediction_train, target_var_flatten).sum(dtype=theano.config.floatX)
corr_eval = categorical_accuracy(prediction_eval, target_var_flatten).sum(dtype=theano.config.floatX)

params = lasagne.layers.get_all_params(layer_output, trainable=True)
updates = lasagne.updates.adadelta(loss_train, params=params)


train_fn = theano.function([input_var, target_var_flatten], [loss_train, corr_train], updates=updates)
eval_fn = theano.function([input_var, target_var_flatten], [loss_eval, corr_eval, final_prediction])

num_batches = num_data / batch_size
lr = learning_rate

for epoch in range(1, num_epochs + 1):
    print 'Epoch %d ...' % epoch
    train_err = 0.0
    train_corr = 0.0
    train_total = 0
    start_time = time.time()
    num_back = 0
    train_batches = 0
    for batch in iterate_minibatches(X_train, y_train, batchsize=batch_size, shuffle=True):
        inputs, targets = batch
        err, corr = train_fn(inputs, targets)
        num = len(inputs)
        train_err += err * num
        train_corr += corr
        train_total += num
        train_batches += 1
        time_ave = (time.time() - start_time) / train_batches
        time_left = (num_batches - train_batches) * time_ave

    print 'train: %d/%d loss: %.4f, acc: %.2f%%, time: %.2fs' % (
        min(train_batches * batch_size, num_data), num_data,
        train_err / train_total, train_corr * 100 / train_total, time.time() - start_time)

    # evaluate performance on dev data
    dev_err = 0.0
    dev_corr = 0.0
    dev_total = 0

    inputs, targets = X_validation, y_validation
    err, corr, predictions = eval_fn(inputs, targets)
    num = len(inputs)
    dev_err += err * num
    dev_corr += corr
    dev_total += num
    print 'dev loss: %.4f, corr: %d, total: %d, acc: %.2f%%' % (
        dev_err / dev_total, dev_corr, dev_total, dev_corr * 100 / dev_total)

    # evaluate on test data when better performance detected
    test_err = 0.0
    test_corr = 0.0
    test_total = 0

    inputs, targets = X_test, y_test
    err, corr, predictions = eval_fn(inputs, targets)
    num = len(inputs)
    test_err += err * num
    test_corr += corr
    test_total += num
    print 'test loss: %.4f, corr: %d, total: %d, acc: %.2f%%' % (
        test_err / test_total, test_corr, test_total, test_corr * 100 / test_total)
