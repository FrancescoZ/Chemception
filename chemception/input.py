import csv
import time
from time import sleep
import collections
import constant
import helpers
from models import Compound

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

import numpy as np

import json
import requests
import time

import random 

extension = 'png'

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


def LoadSMILESData(duplicateProb = 0,seed=7):
	data = LoadData('data',duplicateProb)
	inputs = list(map(lambda x: x.input(t='SMILE'), data))
	with open('./data/elements.txt') as f:
		element_list = f.read().splitlines() 
	dic = buildVocabulary(element_list)
	#dic = buildVocabulary(list(map(lambda x: x._SMILE, data)))

	return np.array(list(map(lambda x: SMILE2Int(x[0],dic),inputs))), np.array(list(map(lambda x: x[1],inputs)))



def readChar(smile):
	chars = []
	for char in smile:
		chars.append(char)
	return chars

def buildVocabulary(dataset):
	words = []
	for data in dataset:
		words.extend(readChar(data))

	counter = collections.Counter(words)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

	words, _ = list(zip(*count_pairs))
	word_to_id = dict(zip(words, range(len(words))))
	
	return word_to_id

def SMILE2Int(smile, vocabularyID):
    data = readChar(smile)
    return [vocabularyID[word] for word in data if word in vocabularyID]