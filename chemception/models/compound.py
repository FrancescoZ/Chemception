from rdkit import Chem
import cv2
import os
import os.path

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

	def input(self, path='',type='image'):
		if type == 'image':
			return self.image(path),1 if self.mutagen else 0
		else:
			return self._SMILE,1 if self.mutagen else 0