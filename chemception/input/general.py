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

def LoadData(fileName,duplicateProb):
	#Molecules array
	compounds = []
	smiles = {}
	start_time = time.time()
	print("Loading Started")
	with open(constant.DATA + fileName + '.csv', newline='') as datasetCsv:
		moleculeReader = csv.reader(datasetCsv, delimiter=';', quotechar=';')
		for i,compound in enumerate(moleculeReader):
			smile = compound[1]
			if smile in smiles and random.random()<duplicateProb:
				continue
			compounds.append(Compound(compound[0],smile,compound[2]=='1'))
			smiles[smile] = 1
	elapsed_time = time.time() - start_time
	print('Load of '+ str(len(compounds))+' finished in '+str(elapsed_time)+'s')
	return compounds