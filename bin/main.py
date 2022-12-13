#!/usr/bin/env python3

from functions import *
from predict import *
from metrics import *
from adapt_data import *
import time

lang = 'fr'
name = 'best_estimator_wthtpunct_upbalanced'
max_depth = 10

print("Testing for "+name+" setting \n")

data, target, test_data, test_target = get_data(lang)

if "wthtpunct" in name.lower() and "lemmatized" in name.lower() :
    data = lemmatize(lang, 'appr', data, 'wthtPunct')
    test_data = lemmatize(lang, 'test', test_data, 'wthtPunct')

elif "wthtpunct" in name.lower() :
    data = remove_punct(data)
    test_data = remove_punct(test_data)

elif "lemmatized" in name.lower():
    data = lemmatize(lang, 'appr', data, '')
    test_data = lemmatize(lang, 'test', test_data, '')

if "balance" in name.lower():
    data, target = balance_data(data, target)
    test_data, test_target = balance_data(test_data, test_target, )

data, test_data = tfidfVectorize(data, test_data)

start = time.time()
pred_target, classes = SGD_classification(name, lang, data, target, test_data)
end = time.time()

print('data reviewed in : '+ str(end-start)+' sec')

print_metrics(test_target, pred_target, classes)