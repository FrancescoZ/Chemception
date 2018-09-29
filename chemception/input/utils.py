import csv
import urllib
import pandas as pd
import time
from time import sleep

from utils import constant
from utils import helpers
from models import Compound
import input as data

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import cirpy
import glob

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools

import argparse 
import os 
import sys 

import json
import requests
import time
import random
import bs4 as bs
import sys
from PyQt5.QtCore import QEventLoop
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWebEngineWidgets import QWebEngineView

def render(source_html):
    """Fully render HTML, JavaScript and all."""

    import sys
    from PyQt5.QtCore import QEventLoop
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtWebEngineWidgets import QWebEngineView

    class Render(QWebEngineView):
        def __init__(self, html):
            self.html = None
            self.app = QApplication(sys.argv)
            QWebEngineView.__init__(self)
            self.loadFinished.connect(self._loadFinished)
            self.setHtml(html)
            while self.html is None:
                self.app.processEvents(QEventLoop.ExcludeUserInputEvents | QEventLoop.ExcludeSocketNotifiers | QEventLoop.WaitForMoreEvents)
            self.app.quit()

        def _callable(self, data):
            self.html = data

        def _loadFinished(self, result):
            self.page().toHtml(self._callable)

    return Render(source_html).html


def CheckData():
	#Molecules array
	compounds = []
	smiles = {}
	start_time = time.time()
	print("Loading Started")
	with open(constant.DATA + 'data' + '.csv', newline='') as datasetCsv:
		moleculeReader = csv.reader(datasetCsv, delimiter=';', quotechar=';')
		for i,compound in enumerate(moleculeReader):
			smile = compound[1]
			if smile in smiles:
				continue
			compounds.append(Compound(compound[0],smile,compound[2]=='1'))
			smiles[smile] = 1
	elapsed_time = time.time() - start_time
	print('Load of '+ str(len(compounds))+' finished in '+str(elapsed_time)+'s')
	for com in compounds:
		if not com.fileExist(constant.IMAGES+"data/"):
			print(com._SMILE)

def ClearData(inputDataset='',outputDataset=''):
	compounds = []
	print('Reading existing database')
	with open(constant.DATA + 'data.csv', newline='') as files:
		data = csv.reader(files, delimiter=';', quotechar=';')
		for i,comp in enumerate(data):
			compounds.append(Compound(comp[0],comp[1],comp[2]=='1'))
	compounds = np.array(compounds)
	with open('cleared_data.csv', 'w', newline='') as files:
		f = csv.writer(files)
		for compound in compounds:
			try:
				(comp, neutralised)= compound.NeutraliseCharges()
				
				s = compound.id+';'+ comp +';'+ str(compound.mutagen)
				print(comp)
				index = np.searchsorted(list(map(lambda x: x._SMILE, compounds)), comp)
				if index< len(compounds) and compounds[index]._SMILE == comp:
					print('Skipped')
					continue
				f.writerow(s)
				print(s)
			except AttributeError as e:
				print(e)
				continue

def CaracterizeData():
	#Loading Data
	compounds = []
	smiles = {}
	start_time = time.time()
	print("Loading Started")
	with open(constant.DATA + 'noduplicates_data.csv', newline='') as datasetCsv:
		moleculeReader = csv.reader(datasetCsv, delimiter=';', quotechar=';')
		for i,compound in enumerate(moleculeReader):
			smile = compound[0]
			if smile in smiles and random.random()<0:
				continue
			compounds.append(Compound(str(i),smile,compound[1]=='1'))
			smiles[smile] = 1
	elapsed_time = time.time() - start_time
	print('Load of '+ str(len(compounds))+' finished in '+str(elapsed_time)+'s')
	
	i = 19321
	helpers.printProgressBar(19321, len(compounds), prefix = 'Progress:', suffix = 'Complete', length = 50)
	with open('data/missing_data_feature.csv', 'w', newline='') as missingFeatures:
		allCompoundMissing = csv.writer(missingFeatures)
		for compound in compounds[19321:len(compounds)]:
			with open('data/data_features.csv', 'a', newline='') as dataFeatures:
				allCompoundFeature = csv.writer(dataFeatures)	
				try:
					start_time = time.time()
					url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/'+ compound._SMILE +'/property/MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES,InChI,IUPACName,XLogP,ExactMass,MonoisotopicMass,TPSA,Complexity,Charge,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,HeavyAtomCount,IsotopeAtomCount,AtomStereoCount,DefinedAtomStereoCount,UndefinedAtomStereoCount,BondStereoCount,DefinedBondStereoCount,UndefinedBondStereoCount,CovalentUnitCount,Volume3D,XStericQuadrupole3D,YStericQuadrupole3D,ZStericQuadrupole3D,FeatureCount3D,FeatureAcceptorCount3D,FeatureDonorCount3D,FeatureAnionCount3D,FeatureCationCount3D,FeatureRingCount3D,FeatureHydrophobeCount3D,ConformerModelRMSD3D,EffectiveRotorCount3D,ConformerCount3D,Fingerprint2D/CSV'
					compoundFeatures = pd.read_csv(url,sep='\n',skiprows=1)
					newRow = compoundFeatures
					allCompoundFeature.writerow(newRow)
					try:
						if elapsed_time <= 0.2:
							time.sleep(0.2-elapsed_time)
						url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/'+compound._SMILE+'/assaysummary/CSV'
						start_time = time.time()
						summary = pd.read_csv(url)
						summary.to_csv('data/dataAssays/'+str(i)+'_assay_summary.csv')
					except:
						print(compound._SMILE+' Assays not Found')
				except urllib.error.HTTPError as err:
					print(compound._SMILE+' Not Found')
					allCompoundMissing.writerow(compound._SMILE)
				i = i+1
				elapsed_time = time.time() - start_time
				if elapsed_time <= 0.2:
					time.sleep(0.2-elapsed_time)
				helpers.printProgressBar(i,len(compounds) , prefix = 'Progress:', suffix = 'Complete', length = 50)

def MergeAssays():
	docs = glob.glob('./data/dataAssays/*.csv')
	with open('data/assays.csv', 'a', newline='') as assays:
		for doc in docs:
			ass = pd.read_csv(doc,sep='\n',skiprows=1)
			ass.to_csv(assays)

