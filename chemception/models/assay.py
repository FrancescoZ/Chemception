from rdkit import Chem
import cv2
import os
import numpy as np
import os.path

class Assay:
	id = ""
	CIDCompound = []
	SIDCompound = []
	result = []
	Description = ""
	ActivityName = []
	Type = ""


	def __init__(self,id,description,ttype):
		self.id=id
		self.Description=description
		self.Type = ttype

	def AddTest(self,CID,SID,Result,Activity):
		self.CIDCompound.append(CID)
		self.SIDCompound.append(SID)
		self.result.append(Result)
		self.ActivityName.append(Activity)
	
