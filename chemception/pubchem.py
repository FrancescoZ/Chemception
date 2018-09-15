import csv
import urllib
import pandas as pd
import time
import argparse 
import os 
import sys 

import json
import requests
import sys

# coding: utf-8
# -*- coding: utf-8 -*-
# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

def LoadPubChem():	
	total = 134725001
	printProgressBar(0, total, prefix = 'Progress:', suffix = 'Complete', length = 50)
	for i in range(total+1):
		with open('/Volume/PubChem/data_features.csv', 'a', newline='') as dataFeatures:
			allCompoundFeature = csv.writer(dataFeatures)
		
			try:
				start_time = time.time()
				url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/CID/'+ str(i) +'/property/MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES,InChI,IUPACName,XLogP,ExactMass,MonoisotopicMass,TPSA,Complexity,Charge,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,HeavyAtomCount,IsotopeAtomCount,AtomStereoCount,DefinedAtomStereoCount,UndefinedAtomStereoCount,BondStereoCount,DefinedBondStereoCount,UndefinedBondStereoCount,CovalentUnitCount,Volume3D,XStericQuadrupole3D,YStericQuadrupole3D,ZStericQuadrupole3D,FeatureCount3D,FeatureAcceptorCount3D,FeatureDonorCount3D,FeatureAnionCount3D,FeatureCationCount3D,FeatureRingCount3D,FeatureHydrophobeCount3D,ConformerModelRMSD3D,EffectiveRotorCount3D,ConformerCount3D,Fingerprint2D/CSV'
				compoundFeatures = pd.read_csv(url,sep='\n',skiprows=1)
				newRow = compoundFeatures
				allCompoundFeature.writerow(newRow)
				try:
					if elapsed_time <= 0.2:
						time.sleep(0.2-elapsed_time)
					url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/CID/'+str(i)+'/assaysummary/CSV'
					start_time = time.time()
					summary = pd.read_csv(url)
					summary.to_csv('/Volume/PubChem/dataAssays/'+str(i)+'_assay_summary.csv')
				except urllib.error.HTTPError as err:
					print(str(i)+' Assays not Found')
			except urllib.error.HTTPError as err:
				print(str(i)+' Not Found')
			elapsed_time = time.time() - start_time
			if elapsed_time <= 0.2:
				time.sleep(0.2-elapsed_time)
			printProgressBar(i,total , prefix = 'Progress:', suffix = 'Complete', length = 50)
