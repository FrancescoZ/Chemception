from network import Chemception
from network import SMILE2Vector 
from network import HATT
import input as data
import helpers
import Optimizer
from evaluation import Metrics

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
import keras.backend as K

import os
import sys
import time
import statistics
import shutil

import numpy as nu

from sklearn.model_selection import train_test_split

#Setting seed to re run the same simulation with the same result
seed = 7
nu.random.seed(seed)

#Defining the size of the network, this can be passed as parameter
N=16
inputSize = 80
#This define which network we want to use:
#	C- Chemception
#	S- SMIELE to Vect
type = 'C'
#Type of network, if none throw error
if len(sys.argv)>1 and sys.argv[1]!=None:
	if sys.argv[1]=='-c' or sys.argv[1] == '-C':
		type='C'
	elif sys.argv[1]=='-s' or sys.argv[1] == '-S':
		type='S'
	elif sys.argv[1]=='-h' or sys.argv[1] == '-H':
		type='H'
	else:
		raise AttributeError("Invalid Network Type")
else:
	raise AttributeError("Network Type missing")
#Execution name, if non will throw an error
if len(sys.argv)>2 and sys.argv[2]!=None:
	if os.path.isdir("./build/"+sys.argv[2]):
		over = input('Execution folder already exist, to you want to overwrite it? [Y/N]')
		if str(over) == 'Y' or str(over)=='y':
			executionName = sys.argv[2]
			shutil.rmtree('./build/'+executionName, ignore_errors=True)
		else:
			raise AttributeError("Execution folder already exists")
	executionName = sys.argv[2]
	os.makedirs('./build/'+executionName)
else: 
	raise AttributeError("Execution name is missing")
#get the size of the simulation if given
loss_function 	= "mean_squared_error"
if len(sys.argv)>3 and sys.argv[3]!=None:
	if sys.argv[3] == 'mse':
		loss_function 	= "mean_squared_error"
	elif sys.argv[3] == 'msle':
		loss_function 	= "mean_squared_logarithmic_error"
	elif sys.argv[3] == 'lc':
		loss_function 	= "logcosh"
	elif sys.argv[3] == 'cc':
		loss_function 	= "categorical_crossentropy"

if len(sys.argv)>4 and sys.argv[4]!=None:
	N=sys.argv[4]
	if len(sys.argv)>5 and sys.argv[5]!=None:
		inputSize=sys.argv[5]

#Setting of the network
batch_size 			= 50
num_classes 		= 2
epochs 				= 10
data_augmentation 	= False
learning_rate		= 1e-3
rho					= 0.9
epsilon				= 1e-8
cross_val			= 2
main_execution_path = './build/'+executionName+'/'
final_resume 		= main_execution_path + '_resume.txt'
# The data, split between train and test sets:
if type =='C':
	(X, Y) 	= data.LoadImageData(extensionImg='png',size=inputSize,duplicateProb=0,seed=seed)
elif type == 'S':
	(X, Y) 	= data.LoadSMILESData(duplicateProb=0,seed=seed)
elif type == 'H':
	(X, Y) 	= data.LoadSMILESData(duplicateProb=0,seed=seed)
cvscores = []
for i in range(2,cross_val+1):

	K.clear_session()
	model_name 						 = type+'_trained_cross_'+str(i)
	current_path 					 = main_execution_path+model_name
	os.makedirs(current_path)
	model_name_file 				 = model_name + '_model.h5'
	model_directory					 = current_path+'/model'
	os.makedirs(model_directory)
	model_path 						 = os.path.join(model_directory, model_name_file)
	log_dir							 = './build/logs/{}'.format(model_name)
	resume_file						 = current_path + '/'+model_name+'_resume.txt'

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1*i, random_state=seed)
	print(X.shape)
	# create model	
	cross_val 						 = cross_val +1	
	x_train 						 = X_train
	#if type=='S':
	#	x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=500, dtype='int32', padding='pre', truncating='pre', value=0)
	#	X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=500, dtype='int32', padding='pre', truncating='pre', value=0)
	y_train 						 = Y_train
	
	# Convert class vectors to binary class matrices.
	if type == 'C':
		y_train 			= keras.utils.to_categorical(y_train, num_classes)
		Y_test 				= keras.utils.to_categorical(Y_test, num_classes)
	tensorBoard = TensorBoard(log_dir=log_dir, 
				histogram_freq=1, 
				batch_size=batch_size, 
				write_graph=False, 
				write_grads=True, 
				write_images=True, 
				embeddings_freq=0, 
				embeddings_layer_names=None, 
				embeddings_metadata=None)
	metrics = Metrics()
	if type =='C':
		model 				= Chemception(N,
									inputSize,
									x_train,
									y_train,
									X_test,
									Y_test,
									learning_rate,
									rho,
									epsilon,
									epochs,
									loss_function,
									log_dir,
									batch_size,
									data_augmentation,
									metrics,
									tensorBoard)
	elif type =='S':
		model 				= SMILE2Vector(N,
									x_train,
									y_train,
									X_test,
									Y_test,
									learning_rate,
									rho,
									epsilon,
									epochs,
									loss_function,
									log_dir,
									batch_size,
									metrics,
									tensorBoard)
	elif type == 'H':
		model 				= HATT( data.char_index,
									x_train,
									y_train,
									X_test,
									Y_test,
									learning_rate,
									rho,
									epsilon,
									epochs,
									loss_function,
									log_dir,
									batch_size,
									metrics,
									tensorBoard)
	#model.print()
	model.run()
	print('Training Ended')
	model.model.save(model_path)
	print('Saved trained model at %s ' % model_path)
	# Score trained model.
	scores = model.model.evaluate(X_test, Y_test, verbose=1)
	print('Test loss:', scores[0])
	print('Test accuracy:', scores[1])
	print('Test precision:', statistics.mean(metrics.precisions))
	print('Test sensitivity:', statistics.mean(metrics.sensitivitys))
	print('Test specificity:', statistics.mean(metrics.specificitys))
	print('Test mcc:', statistics.mean(metrics.mccs))
	print('Test npv:', statistics.mean(metrics.npvs))
	print('Test f1:', statistics.mean(metrics.f1s))

	prec	= statistics.mean(metrics.precisions)
	sens 	= statistics.mean(metrics.sensitivitys)
	spec 	= statistics.mean(metrics.specificitys)
	mcc		= statistics.mean(metrics.mccs)
	npv		= statistics.mean(metrics.npvs)
	f1		= statistics.mean(metrics.f1s)

	f= open(resume_file,"w+")
	f.write('Name:'+ model_name+'\n\n')
	f.write('Test loss:'+ str(scores[0])+'\n')
	f.write('Test accuracy:'+ str(scores[1])+'\n')
	f.write('Test precision:'+ str(prec)+'\n')
	f.write('Test sensitivity:'+ str(sens)+'\n')
	f.write('Test specificity:'+ str(spec)+'\n')
	f.write('Test mcc:'+ str(mcc)+'\n')
	f.write('Test npv:'+ str(npv)+'\n')
	f.write('Test f1:'+ str(f1)+'\n')
	#f.write('Test mcc:', statistics.mean(metrics.val_mccs))
	f.close()

	print('Saved trained resume')
	cvscores.append([scores[0], scores[1], prec , sens, spec, mcc, npv, f1])

cvscores = nu.array(cvscores)
f= open(final_resume,"w+")
f.write('Name:'+ executionName+'\n')
f.write('Loss type:'+ loss_function+'\n\n')
f.write("Total loss: "+str(nu.mean(cvscores[0:len(cvscores),0]))+" (+/- "+str(nu.std(cvscores[0:len(cvscores),0]))+")\n")
f.write("Total accuracy: "+str(nu.mean(cvscores[0:len(cvscores),1]))+" (+/- "+str(nu.std(cvscores[0:len(cvscores),1]))+")\n")
f.write("Total precision: "+str(nu.mean(cvscores[0:len(cvscores),2]))+" (+/- "+str(nu.std(cvscores[0:len(cvscores),2]))+")\n")
f.write("Total sensitivity: "+str(nu.mean(cvscores[0:len(cvscores),3]))+" (+/- "+str(nu.std(cvscores[0:len(cvscores),3]))+")\n")
f.write("Total specificity: "+str(nu.mean(cvscores[0:len(cvscores),4]))+" (+/- "+str(nu.std(cvscores[0:len(cvscores),4]))+")\n")
f.write("Total mcc: "+str(nu.mean(cvscores[0:len(cvscores),5]))+" (+/- "+str(nu.std(cvscores[0:len(cvscores),5]))+")\n")
f.write("Total npv: "+str(nu.mean(cvscores[0:len(cvscores),6]))+" (+/- "+str(nu.std(cvscores[0:len(cvscores),6]))+")\n")
f.write("Total f1: "+str(nu.mean(cvscores[0:len(cvscores),7]))+" (+/- "+str(nu.std(cvscores[0:len(cvscores),7]))+")\n")
#f.write("Total mcc: %.2f%% (+/- %.2f%%)" % (nu.mean(cvscores[5]), nu.std(cvscores[5])))
f.close()

print('Program End')