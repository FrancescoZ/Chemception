from keras import backend as K
from keras.engine.topology import Layer
from network import Chemception

class ChemLayer(Layer):

    def __init__(self, model_weight, **kwargs):
        #self.output_dim = output_dim
        self.model_weight = model_weight
        super(ChemLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.chem = Chemception(16,80,
                                    None,
                                    None,
                                    None,
                                    None,
                                     None,
                                    None,
                                    None, 
                                    None,
                                    None,
                                    None,
                                    None,
                                    None,
                                    None,
                                    None,
                                    None,
									True,
									classes=2).model
        self.chem.load_weights(self.model_weight)
        super(ChemLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return self.chem.predict(x)