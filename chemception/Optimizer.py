from keras.optimizers import SGD
import keras
class OptimizerTracker(keras.callbacks.Callback):

	def on_epoch_end(self, epoch, logs={}):
		learning_rate_init	= 1e-3
		gamma				= 0.92

		lr 			= learning_rate_init * pow(gamma,epoch)
		keras.backend.set_value(self.model.optimizer.lr, lr)
