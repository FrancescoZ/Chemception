import csv
import time
from time import sleep

import constant
import helpers
from models import Compound

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import cirpy

import numpy as np

import json
import requests
import time

extension = 'png'

def LoadData(fileName):
	#Molecules array
	compounds = []
	start_time = time.time()
	print("Loading Started")
	with open(constant.DATA + fileName + '.csv', newline='') as datasetCsv:
		moleculeReader = csv.reader(datasetCsv, delimiter=';', quotechar=';')
		for i,compound in enumerate(moleculeReader):
			compounds.append(Compound(compound[0],compound[1],compound[2]=='1'))
	elapsed_time = time.time() - start_time
	print('Loading Finished in '+str(elapsed_time)+'s')
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

def LoadInput(extensionImg='png',size=80):
	extension = extensionImg
	Compound.extension = extensionImg
	data = LoadData('data')
	#creating input
	if data[0].fileExist(constant.IMAGES+"data/") != True:
		Gen2DImage(data,constant.IMAGES+"data/",size)
	
	inputs = list(map(lambda x: x.input(constant.IMAGES+'data/'), data))
	return np.array(list(map(lambda x: x[0],inputs))), np.array(list(map(lambda x: x[1],inputs)))

def LoadAMES():
	compounds = []
	print('Reading existing database')
	with open(constant.DATA + 'data.csv', newline='') as files:
			data = csv.reader(files, delimiter=';', quotechar=';')
			for i,comp in enumerate(data):
				compounds.append(comp[1])
	compounds = np.array(compounds)
	print('Loading new data')
	suppl = Chem.SDMolSupplier('./AMESdata.sdf')
	with open('new_data.csv', 'w', newline='') as files:
		f = csv.writer(files)
		for compound in suppl:
			smile = str(cirpy.resolve(compound.GetProp('IDNUMBER'), 'smiles'))
			s = compound.GetProp('IDNUMBER')+';'+ smile +';'+compound.GetProp('AMES_Activity')
			index = np.searchsorted(compounds, smile)
			if index< len(compounds) and compounds[index] == smile:
				print('Skipped')
				continue
			f.writerow(s)
			print(s)

def LoadMutagenicity():
	compounds = []
	print('Reading existing database')
	with open(constant.DATA + 'data.csv', newline='') as files:
			data = csv.reader(files, delimiter=';', quotechar=';')
			for i,comp in enumerate(data):
				compounds.append(comp[1])
	compounds = np.array(compounds)
	print('Loading new data')
	suppl = Chem.SmilesMolSupplier('./smiles_cas_N7090.smi')
	with open('new_data copy.csv', 'w', newline='') as files:
		f = csv.writer(files)
		for compound in suppl:
			try:
				smile = str(cirpy.resolve(compound.GetProp('_Name'), 'smiles'))
				s = compound.GetProp('_Name')+';'+ smile +';'+ str(compound.GetProp('0'))
				index = np.searchsorted(compounds, smile)
				if index< len(compounds) and compounds[index] == smile:
					print('Skipped')
					continue
				f.writerow(s)
				print(s)
			except AttributeError as e:
				print(e)
				continue

def DownloadData():
	compounds = []
	with open(constant.DATA + 'data.csv', newline='') as files:
			data = csv.reader(files, delimiter=';', quotechar=';')
			for i,comp in enumerate(data):
				compounds.append(comp[1])
	compounds = np.array(compounds)
	with open(constant.DATA + 'new_data.csv', newline='') as datasetCsv:
		with open(constant.DATA + 'data.csv', 'a', newline='') as files:
			data = csv.reader(datasetCsv, delimiter=';', quotechar=';')
			f = csv.writer(files,)
			for i,compound in enumerate(data):
				try:
					r =  requests.get('https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/'+ compound[0] +'/JSON?heading=Canonical%20SMILES')
					x = json.loads(r.text)
					smile = x['Record']['Section'][0]['Section'][0]['Section'][0]['Information'][0]['StringValue']
					index = np.searchsorted(compounds, smile)
					if index< len(compounds) and compounds[index] == smile:
						continue
					if i % 5 ==0:
						time.sleep(0.5)
						print('Waiting 0.5s')
					if i%300==0:
						time.sleep(30)
						print('Waiting 30s')
					
					r = requests.get('https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/'+ compound[0] +'/JSON?heading=Toxicity')
					x = json.loads(r.text)
					if x['Fault']['Code']  == 'PUGVIEW.NotFound':
						tox = 0
					else:
						tox = 1
					print('PubChem-'+ compound[0]+';'+ smile +';'+str(tox)+';')
					f.writerow('PubChem-'+ compound[0]+';'+ smile +';'+str(tox)+';')
				except ValueError as er:
					print(er)
					continue
