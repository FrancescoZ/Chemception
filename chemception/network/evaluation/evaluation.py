import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import math
from utils import helpers
import time
import csv

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.accs = []
        self.precisions = []
        self.npvs = []
        self.sensitivitys = []
        self.specificitys = []
        self.mccs = []
        self.f1s= []
        self.time = time.time()

    def on_epoch_end(self,epoch, logs={}):
        if not epoch%10 == 0:
            return 
        try:
            print("Other metrics evaluation")
            val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
            val_targ = self.validation_data[1]
            helpers.printProgressBar(0, len(val_predict), prefix = 'Progress:', suffix = 'Complete', length = 50)
            tp =0
            fp = 0
            tn = 0
            fn = 0
            
            for index in range(len(val_predict)):
                if val_targ[index][0] ==1:
                    if val_targ[index][0] == val_predict[index][0]:
                        tp = tp +1
                    else:
                        fn = fn + 1
                else:
                    if val_targ[index][0] == val_predict[index][0]:
                        tn = tn +1
                    else:
                        fp = fp + 1 
                helpers.printProgressBar(index, len(val_predict), prefix = 'Progress:', suffix = 'Complete', length = 50)   

            acc = float(tp + tn)/len(val_predict)
            precision = float(tp)/(tp+ fp + 1e-06)
            npv = float(tn)/(tn + fn + 1e-06)
            sensitivity = float(tp)/ (tp + fn + 1e-06)
            specificity = float(tn)/(tn + fp + 1e-06)
            mcc = float(tp*tn-fp*fn)/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) + 1e-06)
            f1=float(tp*2)/(tp*2+fp+fn+1e-06)
            with open('build-evaluation'+str(self.time)+'.csv',"a+",newline='') as evalCsv:
                evalMetr = csv.writer(evalCsv)
                row= str(epoch)+";"+str(acc)+";"+str(precision)+";"+str(npv)+";"+str(sensitivity)+";"+str(specificity)+";"+str(mcc)+";"+str(f1)
                evalMetr.writerow(row)
            self.accs.append(acc)
            self.precisions.append(precision)
            self.npvs.append(npv)
            self.sensitivitys.append(sensitivity)
            self.specificitys.append(specificity)
            self.mccs.append(mcc)
            self.f1s.append(f1)
			
            logs.update({'mcc': mcc})
            logs.update({'precision': precision})
            logs.update({'npv': npv})
            logs.update({'specificity': specificity})
            logs.update({'sensitivity': sensitivity})
            logs.update({'f1': f1})
            print('val_acc: %f - val_precision: %f - val_npv: %f - val_sensitivity: %f - val_specificitys: %f - val_mcc: %f - val_f1: %f'%( acc, precision, npv,sensitivity,specificity,mcc,f1))
            return
        except ValueError as e:
            print(e)
