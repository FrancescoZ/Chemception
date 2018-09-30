import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from keras.preprocessing.sequence import pad_sequences
import input as data

class Visualizer(object):

    def __init__(self,dic,max_s):
        """
            Visualizes attention maps
            :param padding: the padding to use for the sequences.
            :param input_vocab: the location of the input human
                                vocabulary file
            :param output_vocab: the location of the output 
                                 machine vocabulary file
        """
        self.input_vocab = dic
        self.max_s = max_s
        self.output_vocab = ["0","1"]

    def set_models(self, pred_model, proba_model):
        """
            Sets the models to use
            :param pred_model: the prediction model
            :param proba_model: the model that outputs the activation maps
        """
        self.pred_model = pred_model
        self.proba_model = proba_model

    def attention_map(self, text,path):
        """
            Text to visualze attention map for.
        """
        d = data.SMILE2Int(text,self.input_vocab)
        d = pad_sequences(np.array([d]), maxlen=self.max_s, padding='post')


        # get the output sequence
        predicted_text = "1" if self.pred_model.predict(d).round()[0][0]==1 else "0"
        print(self.pred_model.predict(d).round())
        # get the lengths of the string
        input_length = len(text)
        output_length = 2
        pred = self.proba_model.predict(d)
        predicted = np.zeros((input_length,output_length))
        # get the activation map
        activation_map = np.squeeze(pred[1][0])
        for i in range(0,input_length):
            if predicted_text=='1':
                predicted[i,0]=activation_map[i]
            else:
                predicted[i,1]=activation_map[i]
        #print(predicted)
        predicted = np.rot90(predicted)
        #print(activation_map[0:input_length])

        # import seaborn as sns
        plt.clf()
        f = plt.figure(figsize=(8, 8.5))
        ax = f.add_subplot(1, 1, 1)

        # add image
        i = ax.imshow(predicted, interpolation='nearest', cmap='gray')

        # add colorbar
        cbaxes = f.add_axes([0.2, 0, 0.6, 0.03])
        cbar = f.colorbar(i, cax=cbaxes, orientation='horizontal')
        cbar.ax.set_xlabel('Probability', labelpad=2)

        # add labels
        ax.set_yticks(range(output_length))
        ax.set_yticklabels(['Non-Toxic','Toxic'])

        ax.set_xticks(range(input_length))
        ax.set_xticklabels(text[:input_length], rotation=45)

        ax.set_xlabel('Input Sequence')

        ax.set_ylabel('Output Sequence')

        # add grid and legend
        ax.grid()
        # ax.legend(loc='best')

        #f.savefig(path, bbox_inches='tight')