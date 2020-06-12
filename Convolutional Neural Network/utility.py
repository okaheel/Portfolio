#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import time
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K

EMBEDDING_DIM = 300
numpy.random.seed(137)


def get_best_summary(source='../dataset/data1.txt'):
    item_to_summarize = {}
    for line in open(source, mode='r'):
        parts = line.strip().split('\t')
        item_id = parts[0]
        summary_id = parts[1]
        item_to_summary[item_id] = summary_id

    return item_to_summary


def get_glove_embeddings():
    embeddings_index = {}
    for line in open('../dataset/glove.6B.300d.txt', mode='r'):
        values = line.strip().split()
        word = values[0]
        coefs = numpy.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    return embeddings_index


def get_summary_factors(source ='../dataset/data2.txt'):
    item_factors = {}
    for line in open(source, mode='r'):
        parts = line.strip().split(',')
        item_id_with_first_factor = parts[0]
        item_id = item_id_with_first_factor.split(':')[0]
        first_factor = numpy.asarray(item_id_with_first_factor.split(':')[1], dtype='float32')
        other_factors = numpy.asarray(parts[1:], dtype='float32')
        factors_for_item = numpy.append(first_factor, other_factors)
        item_factors[item_id] = factors_for_item

    return item_factors


def get_embeddings_and_sequences(source='../dataset/processed_summarys_text.txt'):
    start_time = time.time()

    item_factors = get_item_factors()
    summary_texts = []
    summary_factors = numpy.zeros((229901, 20), dtype='float32')
    summarys = []
    summary_index = 0

    for line in open(source, mode='r'):
        parts = line.strip().split('\t')
        item_id = parts[0]
        summary_id = parts[1]

        summarys.append(summary_id)

        summary_factors[summary_index] = item_factors[item_id]
        summary_texts.append(' '.join(parts[2:]))

        summary_index += 1

    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(summary_texts)
    summary_sequences = pad_sequences(tokenizer.texts_to_sequences(summary_texts))
    word_index = tokenizer.word_index

    glove_embeddings = get_glove_embeddings()

    embeddings_matrix = numpy.zeros((len(word_index)+1, EMBEDDING_DIM))
    for word, index in word_index.items():
        embedding_vector = glove_embeddings.get(word)
        if embedding_vector is not None:
            embeddings_matrix[index] = embedding_vector
        else:
            embeddings_matrix[index] = numpy.random.uniform(-0.25, 0.25, EMBEDDING_DIM)

    print('Dataset is read in memory for ' + str(time.time() - start_time) + ' seconds.')

    return embeddings_matrix, summary_sequences, summary_factors, summarys


# splits the dataset in training, test and prediction set
# so that for each item in the training set, 80% of it's summarys are in the training set, and the rest are in the test set.
def get_training_test_and_prediction_set(summarys_sequences, summarys_factors,summarys):
    start_time = time.time()

    train_set_x = []
    train_set_y = []
    test_set_x = []
    test_set_y = []
    prediction_set = []
    prediction_items = []

    training_set_content = [line.strip() for line in open('../dataset/training_set_with_summary_id.txt', mode='r')]
    training_set_length = len(training_set_content)
    # random.shuffle(training_set_content)

    summarys_indexed = {summarys[i]: i for i in range(0, len(summarys))}

    for line in training_set_content:
        training_instance = line.split('\t')
        summary_id = training_instance[3]

        if summary_id in summarys_indexed:
            index = summarys_indexed[summary_id]
            sequence_for_summary = summarys_sequences[index]
            factors_for_summary = summarys_factors[index]

            if len(train_set_x) <= (0.9 * training_set_length):
                train_set_x.append(sequence_for_summary)
                train_set_y.append(factors_for_summary)
            else:
                test_set_x.append(sequence_for_summary)
                test_set_y.append(factors_for_summary)

    items_in_validation_set = set()
    best_summarys_for_items = get_best_summary_for_item()

    for line in open('../dataset/test_set_whole.txt', mode='r'):
        validation_instance = line.strip().split('\t')
        item_id = validation_instance[1]

        if item_id not in items_in_validation_set:
            best_summary_for_item = best_summarys_for_items[item_id]
            summary_index = summarys_indexed[best_summary_for_item]
            sequence_for_summary = summarys_sequences[summary_index]
            prediction_set.append(sequence_for_summary)
            prediction_items.append(item_id)

        items_in_validation_set.add(item_id)

    print('Dataset is split into training, test and prediction for ' + str(time.time() - start_time) + ' seconds.')

    return numpy.array(train_set_x), numpy.array(train_set_y), numpy.array(test_set_x), numpy.array(test_set_y), numpy.array(prediction_set),prediction_items


def get_prediction_set(prediction_indexes, summarys_sequences):
    sequences_to_predict = numpy.zeros(shape=(len(prediction_indexes), len(summarys_sequences[0])), dtype='int32')
    for i in range(0, len(prediction_indexes)):
        sequences_to_predict[i] = summarys_sequences[prediction_indexes[i]]

    return sequences_to_predict


def get_array_elements_as_string(arr):
    array_as_string = []
    for x in numpy.nditer(arr):
        array_as_string.append(str(x))

    return array_as_string


def write_factors_predictions(prediction_items, predictions, epoch):
    factor_predictions_output = open('output/factor_predictions_' + str(epoch) + '.txt', mode='w')
    for i in range(0, len(prediction_items)):

        predictions_for_item = predictions[i]
        factors_output = '\t'.join(get_array_elements_as_string(predictions_for_item))

        item_id = prediction_items[i]

        factor_predictions_output.write(item_id + '\t' + factors_output)
        factor_predictions_output.write('\n')

    factor_predictions_output.flush()
    factor_predictions_output.close()


def RMSE(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))