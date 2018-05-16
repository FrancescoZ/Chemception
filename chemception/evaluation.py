import keras.backend as K
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, matthews_corrcoef

class Metrics(Callback):
	def on_train_begin(self, logs={}):
		self.val_f1s = []
		self.val_recalls = []
		self.val_precisions = []
		self.val_mccs = []
	
	def on_epoch_end(self, epoch, logs={}):
		val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
		val_targ = self.validation_data[1]

		_val_precision,_val_recall, _val_f1,supp  = precision_recall_fscore_support(val_targ, val_predict,average='samples')
		#_val_mcc = matthews_corrcoef(val_targ, val_predict)

		self.val_f1s.append(_val_f1)
		self.val_recalls.append(_val_recall)
		self.val_precisions.append(_val_precision)
		#self.val_mccs.append(_val_mcc)
		print('- val_f1: %f - val_precision: %f - val_recall %f'%( _val_f1, _val_precision, _val_recall))
		return