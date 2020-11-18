#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 3.6

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def remove_symbol(word):
    symbols = [',', '.', '(', ')', ';', ':', '!', '\\n', '\\']
    for symbol in symbols:
        word = word.replace(symbol, '')
    return word

def get_words(message):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    result = []
    for word in word_tokenize(message):
        clean_word = remove_symbol(word)
        stemmed_word = ps.stem(clean_word)
        if not stemmed_word in stop_words:
            result.append(stemmed_word)
    return result

def create_dictionary(messages):
    word_occur_dict = {}
    for message in messages:
        for word in get_words(message):
            if word not in word_occur_dict:
                word_occur_dict[word] = 1
            else:
                word_occur_dict[word] += 1
    word_index_dict = {}
    i = 0
    for word, occur in word_occur_dict.items():
        if occur >= 2:
            word_index_dict[word] = i
            i += 1
    return word_index_dict

def transform_text(messages, word_dictionary):
    mes_len = len(messages)
    word_len = len(word_dictionary)
    matrix = np.zeros((mes_len, word_len))
    for i in range(mes_len):
        for word in get_words(messages[i]):
            if word in word_dictionary:
                matrix[i][word_dictionary[word]] += 1
    return matrix

def fit_naive_bayes_model(matrix, labels):
    # Fit a naive bayes model.
    mes_num, word_count = matrix.shape
    p_1 = np.sum(labels) / mes_num
    phi_1 = np.zeros((word_count)) # Array of num_occurence of each word in message label=1
    phi_0 = np.zeros((word_count))
    num_word_1 = 0
    num_word_0 = 0
    for i, label in enumerate(labels):
        message = matrix[i]
        mes_num = np.sum(message)
        if label == 0:
            num_word_0 += mes_num
        else:
            num_word_1 += mes_num
        for j in range(word_count):
            if label == 0:
                phi_0[j] += matrix[i][j]
            else:
                phi_1[j] += matrix[i][j]
    phi_k0 = np.zeros((word_count))
    phi_k1 = np.zeros((word_count))
    for k in range(word_count):
        phi_k0[k] = (1+phi_0[k]) / (word_count+num_word_0)
        phi_k1[k] = (1+phi_1[k]) / (word_count+num_word_1)
    return ((phi_k0, phi_k1), p_1)

def predict_from_naive_bayes_model(model, matrix):
    prior_1 = model[1]
    prior_0 = 1 - prior_1
    phi_k0, phi_k1 = model[0]
    phi_k0 = np.log(phi_k0)
    phi_k1 = np.log(phi_k1)
    mes_num, word_count = matrix.shape
    pred = np.zeros((mes_num))
    for i in range(mes_num):
        prob_0 = np.log(prior_0)
        prob_1 = np.log(prior_1)
        for j, word_occur in enumerate(matrix[i]):
            prob_0 += word_occur * phi_k0[j]
            prob_1 += word_occur * phi_k1[j]
        if prob_1 > prob_0:
            pred[i] = 1
    return pred

def train_naive_bayes_model(X_train, y_train):
    matrix = transform_text(X_train, create_dictionary(X_train))
    return fit_naive_bayes_model(matrix, y_train)

def predict(model, X_test):
    return predict_from_naive_bayes_model(model, transform_text(X_test, create_dictionary(X_test)))