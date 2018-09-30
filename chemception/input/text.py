import csv
import time
from time import sleep
import collections
from utils import constant
from utils import helpers
from models import Compound

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

import numpy as np

import json
import requests
import time

import random 

import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import keras
from keras.preprocessing.text import Tokenizer
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras.callbacks import Callback
from keras.layers import Embedding, Input, Dense, LSTM, GRU, Bidirectional, TimeDistributed
from keras import backend as K
from keras.models import Model

from sklearn.metrics import roc_auc_score

import input as dataset
import tensorflow as tf

def LoadSMILESData(duplicateProb = 0,seed=7):
    dataComp = dataset.LoadData('data',0)
    smiles = list(map(lambda x: x._SMILE, dataComp))
    tokenizer = Tokenizer(num_words=None, char_level=True)
    tokenizer.fit_on_texts(smiles)
    print(smiles[0])
    dictionary = {}
    i=0
    k=0
    for smile in smiles:
        i+=1
        for c in list(smile):
            k+=1
            if c in dictionary:
                dictionary[c]+=1
            else:
                dictionary[c]=1
    print(len(dictionary))
    # sequence encode
    encoded_docs = tokenizer.texts_to_sequences(smiles)
    # pad sequences
    max_length = max([len(s) for s in smiles])
    vocab = {'C': 1, 'c': 2, '(': 3, ')': 4, 'O': 5, '=': 6, '1': 7, 'N': 8, '2': 9, '3': 10, '[': 11, ']': 12, 'F': 13, '4': 14, 'l': 15, 'n': 16, 'S': 17, '@': 18, 'H': 19, '5': 20, '+': 21, '-': 22, 'B': 23, 'r': 24, '\\': 25, '#': 26, '6': 27, '.': 28, '/': 29, 's': 30, 'P': 31, '7': 32, 'i': 33, 'o': 34, '8': 35, 'I': 36, 'a': 37, '%': 38, '9': 39, '0': 40, 'K': 41, 'e': 42, 'A': 43, 'g': 44, 'p': 45, 'M': 46, 'T': 47, 'b': 48, 'd': 49, 'V': 50, 'Z': 51, 'G': 52, 'L': 53}
    Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    # define vocabulary size (largest integer value)
    labels = list(map(lambda x: 1 if x.mutagen==True else 0,dataComp))
    return Xtrain, labels,vocab,max_length



def readChar(smile):
    chars = []
    for char in smile:
        chars.append(char)
    return chars

def SMILE2Int(smile, vocabularyID):
    data = readChar(smile)
    return [vocabularyID[word] for word in data if word in vocabularyID]