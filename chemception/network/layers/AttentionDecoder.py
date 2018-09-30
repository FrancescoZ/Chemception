
# import tensorflow as tf
# from keras import backend as K
# from keras import regularizers, constraints, initializers, activations
# from keras.layers.recurrent import Recurrent
# from keras.engine import InputSpec
# from network.layers.TimeDistributedDens import *

# tfPrint = lambda d, T: tf.Print(input_=T, data=[T, tf.shape(T)], message=d)

# class AttentionDecoder(Recurrent):

#     def __init__(self, units, output_dim,
#                  activation='tanh',
#                  return_probabilities=False,
#                  name='AttentionDecoder',
#                  kernel_initializer='glorot_uniform',
#                  recurrent_initializer='orthogonal',
#                  bias_initializer='zeros',
#                  kernel_regularizer=None,
#                  bias_regularizer=None,
#                  activity_regularizer=None,
#                  kernel_constraint=None,
#                  bias_constraint=None,
#                  **kwargs):
#         """
#         Implements an AttentionDecoder that takes in a sequence encoded by an
#         encoder and outputs the decoded states 
#         :param units: dimension of the hidden state and the attention matrices
#         :param output_dim: the number of labels in the output space

#         references:
#             Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. 
#             "Neural machine translation by jointly learning to align and translate." 
#             arXiv preprint arXiv:1409.0473 (2014).
#         """
#         self.units = units
#         self.output_dim = output_dim
#         self.return_probabilities = return_probabilities
#         self.activation = activations.get(activation)
#         self.kernel_initializer = initializers.get(kernel_initializer)
#         self.recurrent_initializer = initializers.get(recurrent_initializer)
#         self.bias_initializer = initializers.get(bias_initializer)

#         self.kernel_regularizer = regularizers.get(kernel_regularizer)
#         self.recurrent_regularizer = regularizers.get(kernel_regularizer)
#         self.bias_regularizer = regularizers.get(bias_regularizer)
#         self.activity_regularizer = regularizers.get(activity_regularizer)

#         self.kernel_constraint = constraints.get(kernel_constraint)
#         self.recurrent_constraint = constraints.get(kernel_constraint)
#         self.bias_constraint = constraints.get(bias_constraint)

#         super(AttentionDecoder, self).__init__(**kwargs)
#         self.name = name
#         self.return_sequences = True  # must return sequences

#     def build(self, input_shape):
#         """
#           See Appendix 2 of Bahdanau 2014, arXiv:1409.0473
#           for model details that correspond to the matrices here.
#         """

#         self.batch_size, self.timesteps, self.input_dim = input_shape

#         if self.stateful:
#             super(AttentionDecoder, self).reset_states()

#         self.states = [None, None]  # y, s

#         """
#             Matrices for creating the context vector
#         """

#         self.V_a = self.add_weight(shape=(self.units,),
#                                    name='V_a',
#                                    initializer=self.kernel_initializer,
#                                    regularizer=self.kernel_regularizer,
#                                    constraint=self.kernel_constraint)
#         self.W_a = self.add_weight(shape=(self.units, self.units),
#                                    name='W_a',
#                                    initializer=self.kernel_initializer,
#                                    regularizer=self.kernel_regularizer,
#                                    constraint=self.kernel_constraint)
#         self.U_a = self.add_weight(shape=(self.input_dim, self.units),
#                                    name='U_a',
#                                    initializer=self.kernel_initializer,
#                                    regularizer=self.kernel_regularizer,
#                                    constraint=self.kernel_constraint)
#         self.b_a = self.add_weight(shape=(self.units,),
#                                    name='b_a',
#                                    initializer=self.bias_initializer,
#                                    regularizer=self.bias_regularizer,
#                                    constraint=self.bias_constraint)
#         """
#             Matrices for the r (reset) gate
#         """
#         self.C_r = self.add_weight(shape=(self.input_dim, self.units),
#                                    name='C_r',
#                                    initializer=self.recurrent_initializer,
#                                    regularizer=self.recurrent_regularizer,
#                                    constraint=self.recurrent_constraint)
#         self.U_r = self.add_weight(shape=(self.units, self.units),
#                                    name='U_r',
#                                    initializer=self.recurrent_initializer,
#                                    regularizer=self.recurrent_regularizer,
#                                    constraint=self.recurrent_constraint)
#         self.W_r = self.add_weight(shape=(self.output_dim, self.units),
#                                    name='W_r',
#                                    initializer=self.recurrent_initializer,
#                                    regularizer=self.recurrent_regularizer,
#                                    constraint=self.recurrent_constraint)
#         self.b_r = self.add_weight(shape=(self.units, ),
#                                    name='b_r',
#                                    initializer=self.bias_initializer,
#                                    regularizer=self.bias_regularizer,
#                                    constraint=self.bias_constraint)

#         """
#             Matrices for the z (update) gate
#         """
#         self.C_z = self.add_weight(shape=(self.input_dim, self.units),
#                                    name='C_z',
#                                    initializer=self.recurrent_initializer,
#                                    regularizer=self.recurrent_regularizer,
#                                    constraint=self.recurrent_constraint)
#         self.U_z = self.add_weight(shape=(self.units, self.units),
#                                    name='U_z',
#                                    initializer=self.recurrent_initializer,
#                                    regularizer=self.recurrent_regularizer,
#                                    constraint=self.recurrent_constraint)
#         self.W_z = self.add_weight(shape=(self.output_dim, self.units),
#                                    name='W_z',
#                                    initializer=self.recurrent_initializer,
#                                    regularizer=self.recurrent_regularizer,
#                                    constraint=self.recurrent_constraint)
#         self.b_z = self.add_weight(shape=(self.units, ),
#                                    name='b_z',
#                                    initializer=self.bias_initializer,
#                                    regularizer=self.bias_regularizer,
#                                    constraint=self.bias_constraint)
#         """
#             Matrices for the proposal
#         """
#         self.C_p = self.add_weight(shape=(self.input_dim, self.units),
#                                    name='C_p',
#                                    initializer=self.recurrent_initializer,
#                                    regularizer=self.recurrent_regularizer,
#                                    constraint=self.recurrent_constraint)
#         self.U_p = self.add_weight(shape=(self.units, self.units),
#                                    name='U_p',
#                                    initializer=self.recurrent_initializer,
#                                    regularizer=self.recurrent_regularizer,
#                                    constraint=self.recurrent_constraint)
#         self.W_p = self.add_weight(shape=(self.output_dim, self.units),
#                                    name='W_p',
#                                    initializer=self.recurrent_initializer,
#                                    regularizer=self.recurrent_regularizer,
#                                    constraint=self.recurrent_constraint)
#         self.b_p = self.add_weight(shape=(self.units, ),
#                                    name='b_p',
#                                    initializer=self.bias_initializer,
#                                    regularizer=self.bias_regularizer,
#                                    constraint=self.bias_constraint)
#         """
#             Matrices for making the final prediction vector
#         """
#         self.C_o = self.add_weight(shape=(self.input_dim, self.output_dim),
#                                    name='C_o',
#                                    initializer=self.recurrent_initializer,
#                                    regularizer=self.recurrent_regularizer,
#                                    constraint=self.recurrent_constraint)
#         self.U_o = self.add_weight(shape=(self.units, self.output_dim),
#                                    name='U_o',
#                                    initializer=self.recurrent_initializer,
#                                    regularizer=self.recurrent_regularizer,
#                                    constraint=self.recurrent_constraint)
#         self.W_o = self.add_weight(shape=(self.output_dim, self.output_dim),
#                                    name='W_o',
#                                    initializer=self.recurrent_initializer,
#                                    regularizer=self.recurrent_regularizer,
#                                    constraint=self.recurrent_constraint)
#         self.b_o = self.add_weight(shape=(self.output_dim, ),
#                                    name='b_o',
#                                    initializer=self.bias_initializer,
#                                    regularizer=self.bias_regularizer,
#                                    constraint=self.bias_constraint)

#         # For creating the initial state:
#         self.W_s = self.add_weight(shape=(self.input_dim, self.units),
#                                    name='W_s',
#                                    initializer=self.recurrent_initializer,
#                                    regularizer=self.recurrent_regularizer,
#                                    constraint=self.recurrent_constraint)

#         self.input_spec = [
#             InputSpec(shape=(self.batch_size, self.timesteps, self.input_dim))]
#         self.built = True

#     def call(self, x):
#         # store the whole sequence so we can "attend" to it at each timestep
#         self.x_seq = x

#         # apply the a dense layer over the time dimension of the sequence
#         # do it here because it doesn't depend on any previous steps
#         # thefore we can save computation time:
#         self._uxpb = time_distributed_dense(self.x_seq, self.U_a, b=self.b_a,
#                                              input_dim=self.input_dim,
#                                              timesteps=self.timesteps,
#                                              output_dim=self.units)

#         return super(AttentionDecoder, self).call(x)

#     def get_initial_state(self, inputs):
#         print('inputs shape:', inputs.get_shape())

#         # apply the matrix on the first time step to get the initial s0.
#         s0 = activations.tanh(K.dot(inputs[:, 0], self.W_s))

#         # from keras.layers.recurrent to initialize a vector of (batchsize,
#         # output_dim)
#         y0 = K.zeros_like(inputs)  # (samples, timesteps, input_dims)
#         y0 = K.sum(y0, axis=(1, 2))  # (samples, )
#         y0 = K.expand_dims(y0)  # (samples, 1)
#         y0 = K.tile(y0, [1, self.output_dim])

#         return [y0, s0]


#     def compute_output_shape(self, input_shape):
#         """
#             For Keras internal compatability checking
#         """
#         if self.return_probabilities:
#             return (None, self.timesteps, self.timesteps)
#         else:
#             return (None, self.timesteps, self.output_dim)

#     def get_config(self):
#         """
#             For rebuilding models on load time.
#         """
#         config = {
#             'output_dim': self.output_dim,
#             'units': self.units,
#             'return_probabilities': self.return_probabilities
#         }
#         base_config = super(AttentionDecoder, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
###########################################################################################
###########################################################################################
###########################################################################################
# import tensorflow as tf
# from keras import backend as K
# from keras import regularizers, constraints, initializers, activations
# from keras.layers.recurrent import Recurrent
# from keras.engine import InputSpec
# from network.layers.TimeDistributedDens import *

# tfPrint = lambda d, T: tf.Print(input_=T, data=[T, tf.shape(T)], message=d)

# class AttentionDecoder(Recurrent):

#     def __init__(self, units, output_dim,
#                  activation='tanh',
#                  return_probabilities=False,
#                  name='AttentionDecoder',
#                  kernel_initializer='glorot_uniform',
#                  recurrent_initializer='orthogonal',
#                  bias_initializer='zeros',
#                  kernel_regularizer=None,
#                  bias_regularizer=None,
#                  activity_regularizer=None,
#                  kernel_constraint=None,
#                  bias_constraint=None,
#                  **kwargs):
#         """
#         Implements an AttentionDecoder that takes in a sequence encoded by an
#         encoder and outputs the decoded states 
#         :param units: dimension of the hidden state and the attention matrices
#         :param output_dim: the number of labels in the output space

#         references:
#             Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. 
#             "Neural machine translation by jointly learning to align and translate." 
#             arXiv preprint arXiv:1409.0473 (2014).
#         """
#         self.init = initializers.get('normal')
#         self.supports_masking = True
#         self.attention_dim = units
        
#         self.activation = activation
#         self.kernel_initializer = kernel_initializer
#         self.recurrent_initializer= recurrent_initializer
#         self.bias_initializer = bias_initializer
#         self.kernel_regularizer= kernel_regularizer
#         self.bias_regularizer = bias_regularizer
#         self.activity_regularizer=activity_regularizer
#         self.kernel_constraint= kernel_constraint
#         self.bias_constraint=bias_constraint
#         super(AttentionDecoder, self).__init__(**kwargs)
#         self.name = name
#         self.return_sequences = True  # must return sequences
#         self.return_probabilities = return_probabilities
#         self.output_dim = output_dim
        



#     def build(self, input_shape):
#         assert len(input_shape) == 3
#         self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
#         self.b = K.variable(self.init((self.attention_dim, )))
#         self.u = K.variable(self.init((self.attention_dim, 1)))
#         self.trainable_weights = [self.W, self.b, self.u]
#         self.built = True
#         super(AttentionDecoder, self).build(input_shape)

#     def call(self, x,mask=None):
#         # size of x :[batch_size, sel_len, attention_dim]
#         # size of u :[batch_size, attention_dim]
#         # uit = tanh(xW+b)
#         uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
#         ait = K.dot(uit, self.u)
#         ait = K.squeeze(ait, -1)

#         ait = K.exp(ait)
#         if mask is not None:
#             # Cast the mask to floatX to avoid float64 upcasting in theano
#             ait *= K.cast(mask, K.floatx())
#         ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
#         ait = K.expand_dims(ait)
#         weighted_input = x * ait
#         output = K.sum(weighted_input, axis=1)

#         return output

#     def get_initial_state(self, inputs):
#         print('inputs shape:', inputs.get_shape())

#         # apply the matrix on the first time step to get the initial s0.
#         s0 = activations.tanh(K.dot(inputs[:, 0], self.W_s))

#         # from keras.layers.recurrent to initialize a vector of (batchsize,
#         # output_dim)
#         y0 = K.zeros_like(inputs)  # (samples, timesteps, input_dims)
#         y0 = K.sum(y0, axis=(1, 2))  # (samples, )
#         y0 = K.expand_dims(y0)  # (samples, 1)
#         y0 = K.tile(y0, [1, self.output_dim])

#         return [y0, s0]

#     def compute_mask(self, inputs, mask=None):
#         output_mask = mask if self.return_probabilities else None
#         # do not pass the mask to the next layers
#         return output_mask

#     def compute_output_shape(self, input_shape):
#         """
#             For Keras internal compatability checking
#         """
#         return (input_shape[0], input_shape[-1])

#     def get_config(self):
#         """
#             For rebuilding models on load time.
#         """
#         config = {
#             'output_dim': self.output_dim,
#             'units': self.attention_dim,
#             'attention_dim': self.attention_dim,
#             'return_probabilities': self.return_probabilities
#         }
#         base_config = super(AttentionDecoder, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
###########################################################################################
###########################################################################################
###########################################################################################
from keras import backend as K, initializers, regularizers, constraints
from keras.engine.topology import Layer
from keras.layers.core import Layer  
from keras import regularizers, constraints  
from keras import backend as K

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        # todo: check that this is correct
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class AttentionDecoder(Layer):
    def __init__(self,
                 units = 1,
                 output_dim = 1,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True,
                 return_sequence = True,
                 return_attention=False,
                 return_probabilities = False,
                 **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Note: The layer has been tested with Keras 1.x
        Example:
        
            # 1
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...
            # 2 - Get the attention scores
            hidden = LSTM(64, return_sequences=True)(words)
            sentence, word_scores = Attention(return_attention=True)(hidden)
        """
        self.supports_masking = True
        self.return_attention = return_attention
        self.init = initializers.get('glorot_uniform')
        self.attention_dim = units
        self.output_dim = output_dim
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionDecoder, self).__init__(**kwargs)

    def build(self, input_shape):
        #assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        eij = dot_product(x, self.W)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        weighted_input = x * K.expand_dims(a)

        result = K.sum(weighted_input, axis=1)

        if self.return_attention:
            return [result, a]
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]
    
    def get_config(self):
        """
            For rebuilding models on load time.
        """
        config = {
            'output_dim': self.output_dim,
            'units': self.attention_dim,
            'attention_dim': self.attention_dim,
            'return_probabilities': self.return_attention
        }
        base_config = super(AttentionDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
