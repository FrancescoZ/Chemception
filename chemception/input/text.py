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
	Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
	# define vocabulary size (largest integer value)
	labels = list(map(lambda x: 1 if x.mutagen==True else 0,dataComp))
	return Xtrain, labels,tokenizer.word_index,max_length



def readChar(smile):
	chars = []
	for char in smile:
		chars.append(char)
	return chars

def SMILE2Int(smile, vocabularyID):
    data = readChar(smile)
    return [vocabularyID[word] for word in data if word in vocabularyID]