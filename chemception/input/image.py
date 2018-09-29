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

from input import LoadData

extension = 'png'

def Gen2DImage(compounds,path,size):
	l = len(compounds)
	print('Generating 2D images structure in '+ extension)
	helpers.printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
	start_time = time.time()
	for i, comp in enumerate(compounds): 
		#Creating coordinates
		tmp = AllChem.Compute2DCoords(comp.rdkMolecule)
		#drawing image
		Draw.MolToFile(comp.rdkMolecule,path + comp.id+'.'+extension, size=(size, size))
		# Update Progress Bar
		helpers.printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
	elapsed_time = time.time() - start_time
	print('Image generation finished in '+str(elapsed_time)+'s')

#def Gen3DImage(compounds,path):
	# l = len(compounds)
	# print('Generating 3D images structure')
	# helpers.printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
	# start_time = time.time()
	# for i, comp in enumerate(compounds): 
	# 	#Creating coordinates
	# 	tmp = AllChem.Compute2DCoords(comp.rdkMolecule)
	# 	#drawing image
	# 	Draw.MolToFile(comp.rdkMolecule,path + comp.id+'.png', size=(80, 80))
	# 	# Update Progress Bar
	# 	helpers.printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
	# elapsed_time = time.time() - start_time
	# print('Image generation finished in '+str(elapsed_time)+'s')


def LoadImageData(extensionImg='png',size=80, duplicateProb = 0, seed = 7):
	extension = extensionImg
	Compound.extension = extensionImg
	random.seed(seed)
	data = LoadData('data',duplicateProb)
	#creating input
	if data[0].fileExist(constant.IMAGES+"data/") != True:
		Gen2DImage(data,constant.IMAGES+"data/",size)
	inputs = list(map(lambda x: x.input(constant.IMAGES+'data/',t='image'), data))
	return np.array(list(map(lambda x: x[0],inputs))), np.array(list(map(lambda x: x[1],inputs)))