#!/usr/bin/env python3

from functions import *
from predict import *
from metrics import *
from adapt_data import *
import time

### On set les paramètres que l'on souhaite tester ###
lang = 'it'
name = ''
max_depth = 10
######################################################

print("Testing for "+name+" setting \n")

data, target, test_data, test_target = get_data(lang)

### On verifie les paramètres des données que l'on souhaite évaluer ###
if "wthtpunct" in name.lower() and "lemmatize" in name.lower() :
    data = lemmatize(lang, 'appr', data, 'wthtPunct')
    test_data = lemmatize(lang, 'test', test_data, 'wthtPunct')

elif "wthtpunct" in name.lower() :
    data = remove_punct(data)
    test_data = remove_punct(test_data)

elif "lemmatize" in name.lower():
    data = lemmatize(lang, 'appr', data, '')
    test_data = lemmatize(lang, 'test', test_data, '')

if "balance" in name.lower():
    data, target = balance_data(data, target)
    test_data, test_target = balance_data(test_data, test_target, )
########################################################################

data, test_data = tfidfVectorize(data, test_data)

### On entraîne le modèle en retenant le temps d'entraînement ###
start = time.time()
pred_target, classes = random_forest(name, lang, data, target, test_data)
end = time.time()
##################################################################

### On imprime les résultats ###
print('data reviewed in : '+ str(end-start)+' sec')
print_metrics(test_target, pred_target, classes)
################################