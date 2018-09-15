from rdkit import Chem
import cv2
import os
import numpy as np
import os.path

from rdkit import Chem
from rdkit.Chem import AllChem

class Compound:
	extension = 'png'
	#compound identifier in the dataset
	id=""
	#compound SMILE
	_SMILE=""
	#mutagen
	mutagen=False
	#rdk model
	rdkMolecule = None

	def __init__(self,id,SMILE,mut):
		self.id=id
		self._SMILE=SMILE
		self.description = self.id + ": "+ self._SMILE
		self.mutagen = mut
		self.rdkMolecule = Chem.MolFromSmiles(self._SMILE)
		#print(SMILE)

	def __repr__(self):
		return self.description
	def __str__(self):
		return self.description

	def fileExist(self,path):
		img = path+self.id+'.'+ Compound.extension
		return os.path.isfile(img)

	def image(self,path):
		img = path+self.id+'.'+Compound.extension
		return cv2.imread(str(img))

	def input(self, path='',t='image'):
		if t == 'image':
			return self.image(path),1 if self.mutagen else 0
		else:
			return self._SMILE,1 if self.mutagen else 0
	
	def InitialiseNeutralisationReactions(self):
		patts= (
			# Imidazoles
			('[n+;H]','n'),
			# Amines
			('[N+;!H0]','N'),
			# Carboxylic acids and alcohols
			('[$([O-]);!$([O-][#7])]','O'),
			# Thiols
			('[S-;X1]','S'),
			# Sulfonamides
			('[$([N-;X2]S(=O)=O)]','N'),
			# Enamines
			('[$([N-;X2][C,N]=C)]','N'),
			# Tetrazoles
			('[n-]','[nH]'),
			# Sulfoxides
			('[$([S-]=O)]','S'),
			# Amides
			('[$([N-]C=O)]','N'),
			)
		return [(Chem.MolFromSmarts(x),Chem.MolFromSmiles(y,False)) for x,y in patts]

	reac=None
	def NeutraliseCharges(self, reactions=None):
		if reactions is None:
			if self.reac is None:
				self.reac=self.InitialiseNeutralisationReactions()
			reactions=self.reac
		mol = Chem.MolFromSmiles(self._SMILE)
		replaced = False
		for i,(reactant, product) in enumerate(reactions):
			while mol.HasSubstructMatch(reactant):
				replaced = True
				rms = AllChem.ReplaceSubstructs(mol, reactant, product)
				mol = rms[0]
		if replaced:
			return (Chem.MolToSmiles(mol,True), True)
		else:
			return (self._SMILE, False)
