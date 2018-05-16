import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

class Metrics(Callback):
	def on_train_begin(self, logs={}):
		self.val_f1s = []
		self.val_recalls = []
		self.val_precisions = []
		self.val_mccs = []
	
	def on_epoch_end(self, epoch, logs={}):
		try:
			val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
			val_targ = self.validation_data[1]

			_val_f1 = f1_score(val_targ, val_predict,average='micro')
			print('f1')
			_val_recall = recall_score(val_targ, val_predict,average='micro')
			print('recall')
			_val_precision = precision_score(val_targ, val_predict,average='micro')
			print('prec')
			#_val_mcc = matthews_corrcoef(val_targ, val_predict)

			self.val_f1s.append(_val_f1)
			self.val_recalls.append(_val_recall)
			self.val_precisions.append(_val_precision)
			#self.val_mccs.append(_val_mcc)
			print('- val_f1: %f - val_precision: %f - val_recall %f'%( _val_f1, _val_precision, _val_recall))
			return
		except ValueError as e:
			print((np.asarray(self.model.predict(self.validation_data[0]))).round())
			print(self.validation_data[1])
