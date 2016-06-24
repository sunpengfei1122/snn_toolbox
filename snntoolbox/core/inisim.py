# -*- coding: utf-8 -*-
"""
INI spiking neuron simulator

A collection of helper functions, including spiking layer classes derived from
Keras layers, which were used to implement our own IF spiking simulator.

Not needed when converting and running the SNN in pyNN.

Created on Tue Dec  8 10:41:10 2015

@author: rbodo
"""

# For compatibility with python2
from builtins import super

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.shared_randomstreams import RandomStreams
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import AveragePooling2D, Convolution2D
from keras import backend as K
from snntoolbox.config import settings

rng = RandomStreams()


def floatX(X):
    return [np.asarray(x, dtype=theano.config.floatX) for x in X]


def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)


def shared_zeros(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.zeros(shape), dtype=dtype, name=name)


def on_gpu():
    return theano.config.device[:3] == 'gpu'

if on_gpu():
    from theano.sandbox.cuda import dnn


def update_neurons(self, impulse, time, updates):
    if 'activation' in self.get_config() and \
            self.get_config()['activation'] == 'softmax':
        return softmax_activation(self, impulse, time, updates)
    return linear_activation(self, impulse, time, updates)


def linear_activation(self, impulse, time, updates):
    # Destroy impulse if in refractory period
    masked_imp = T.set_subtensor(
        impulse[(self.refrac_until > time).nonzero()], 0.)
    # Add impulse
    new_mem = self.mem + masked_imp
    # Store spiking
    output_spikes = new_mem > self.v_thresh
    # At spike, reduce membrane potential by one instead of resetting to zero,
    # so that no information stored in membrane potential is lost. This reduces
    # the variance in the spikerate-activation correlation plot for activations
    # greater than 0.5.
    new_and_reset_mem = T.inc_subtensor(new_mem[output_spikes.nonzero()], -1.)
    # Alternatively, perform standard reset:
    # new_and_reset_mem = T.set_subtensor(new_mem[output_spikes.nonzero()], 0.)
    # Store refractory
    new_refractory = T.set_subtensor(
        self.refrac_until[output_spikes.nonzero()], time + self.tau_refrac)
    updates.append((self.refrac_until, new_refractory))
    updates.append((self.mem, new_and_reset_mem))
    updates.append((self.spiketrain, output_spikes * time))
    return output_spikes


def softmax_activation(self, impulse, time, updates):
    # Destroy impulse if in refractory period
    masked_imp = T.set_subtensor(
        impulse[(self.refrac_until > time).nonzero()], 0.)
    # Add impulse
    new_mem = self.mem + masked_imp
    # Store spiking
    output_spikes, new_and_reset_mem = theano.ifelse.ifelse(
        T.le(rng.uniform(), 100 * settings['dt'] / 1000),  # Ext. Poisson clock
        trigger_spike(new_mem), skip_spike(new_mem))  # Then and else condition
    # Store refractory. In case of a spike we are resetting all neurons, even
    # the ones that didn't spike. However, in the refractory period we only
    # consider those that spiked. May have to change that...
    new_refractory = T.set_subtensor(
        self.refrac_until[output_spikes.nonzero()], time + self.tau_refrac)
    updates.append((self.refrac_until, new_refractory))
    updates.append((self.mem, new_and_reset_mem))
    updates.append((self.spiketrain, output_spikes * time))
    return output_spikes


def trigger_spike(new_mem):
    activ = T.nnet.softmax(new_mem)
    max_activ = T.max(activ, axis=1, keepdims=True)
    output_spikes = T.eq(activ, max_activ).astype('float32')
    return output_spikes, T.zeros_like(new_mem)


def skip_spike(new_mem):
    return T.zeros_like(new_mem), new_mem


def reset(self):
    if self.inbound_nodes[0].inbound_layers:
        reset(self.inbound_nodes[0].inbound_layers[0])
    self.mem.set_value(floatX(np.zeros(self.output_shape)))
    self.refrac_until.set_value(floatX(np.zeros(self.output_shape)))


def get_input(self):
    if self.inbound_nodes[0].inbound_layers:
        if 'input' in self.inbound_nodes[0].inbound_layers[0].name:
            previous_output = self.input
        else:
            previous_output = \
                self.inbound_nodes[0].inbound_layers[0].get_output()
    else:
        previous_output = K.placeholder(shape=self.input_shape)
    return previous_output, get_time(self), get_updates(self)


def get_time(self):
    if hasattr(self, 'time_var'):
        return self.time_var
    elif self.inbound_nodes[0].inbound_layers:
        return get_time(self.inbound_nodes[0].inbound_layers[0])
    else:
        raise Exception("Layer is not connected and is not an input layer.")


def get_updates(self):
    if self.inbound_nodes[0].inbound_layers:
        return self.inbound_nodes[0].inbound_layers[0].updates
    else:
        return []


def init_neurons(self, v_thresh=1.0, tau_refrac=0.0, **kwargs):
    # The neurons in the spiking layer cannot be initialized until the layer
    # has been initialized and connected to the network. Otherwise
    # 'output_shape' is not known (obtained from previous layer), and
    # the 'input' attribute will not be overwritten by the layer's __init__.
    self.v_thresh = v_thresh
    self.tau_refrac = tau_refrac
    self.refrac_until = shared_zeros(self.output_shape)
    self.mem = shared_zeros(self.output_shape)
    self.spiketrain = shared_zeros(self.output_shape)
    self.updates = []
    if 'time_var' in kwargs:
        input_layer = self.inbound_nodes[0].inbound_layers[0]
        input_layer.v_thresh = v_thresh
        input_layer.tau_refrac = tau_refrac
        input_layer.refrac_until = shared_zeros(self.output_shape)
        input_layer.time_var = kwargs['time_var']
        input_layer.mem = shared_zeros(self.output_shape)
        input_layer.spiketrain = shared_zeros(self.output_shape)
        input_layer.updates = []


class SpikeFlatten(Flatten):
    def __init__(self, label=None, **kwargs):
        super().__init__(**kwargs)
        if label is not None:
            self.label = label
        else:
            self.label = self.name

    def get_output(self, train=False):
        # Recurse
        inp, time, updates = get_input(self)
        self.updates = updates
        reshaped_inp = T.reshape(inp, self.output_shape)
        return reshaped_inp


class SpikeDense(Dense):
    """ batch_size x input_shape x out_shape """
    def __init__(self, output_dim, weights=None, label=None, **kwargs):
        super().__init__(output_dim, weights=weights, **kwargs)
        if label is not None:
            self.label = label
        else:
            self.label = self.name

    def get_output(self, train=False):
        # Recurse
        inp, time, updates = get_input(self)
        # Get impulse
        self.impulse = T.add(T.dot(inp, self.get_weights()[0]),
                             self.get_weights()[1])
        output_spikes = update_neurons(self, self.impulse, time, updates)
        self.updates = updates
        return T.cast(output_spikes, 'float32')


class SpikeConv2DReLU(Convolution2D):
    """ batch_size x input_shape x out_shape """
    def __init__(self, nb_filter, nb_row, nb_col, weights=None,
                 border_mode='valid', subsample=(1, 1), label=None,
                 filter_flip=True, **kwargs):
        super().__init__(nb_filter, nb_row, nb_col, weights=weights,
                         border_mode=border_mode, subsample=subsample,
                         **kwargs)
        self.filter_flip = filter_flip
        if label is not None:
            self.label = label
        else:
            self.label = self.name

    def get_output(self, train=False):
        # Recurse
        inp, time, updates = get_input(self)

        # CALCULATE SYNAPTIC SUMMED INPUT
        border_mode = self.border_mode
        if on_gpu() and dnn.dnn_available():
            conv_mode = 'conv' if self.filter_flip else 'cross'
            if border_mode == 'same':
                assert(self.subsample == (1, 1))
                pad_x = (self.nb_row - self.subsample[0]) // 2
                pad_y = (self.nb_col - self.subsample[1]) // 2
                conv_out = dnn.dnn_conv(img=inp,
                                        kerns=self.get_weights()[0],
                                        border_mode=(pad_x, pad_y),
                                        conv_mode=conv_mode)
            else:
                conv_out = dnn.dnn_conv(img=inp,
                                        kerns=self.get_weights()[0],
                                        border_mode=border_mode,
                                        subsample=self.subsample,
                                        conv_mode=conv_mode)
        else:
            if border_mode == 'same':
                border_mode = 'full'
            conv_out = T.nnet.conv2d(inp, self.get_weights()[0],
                                     border_mode=border_mode,
                                     subsample=self.subsample,
                                     filter_flip=self.filter_flip)
            if self.border_mode == 'same':
                shift_x = (self.nb_row - 1) // 2
                shift_y = (self.nb_col - 1) // 2
                conv_out = conv_out[:, :, shift_x:inp.shape[2] + shift_x,
                                    shift_y:inp.shape[3] + shift_y]

        self.impulse = conv_out + K.reshape(self.get_weights()[1],
                                            (1, self.nb_filter, 1, 1))
        output_spikes = update_neurons(self, self.impulse, time, updates)
        self.updates = updates
        return T.cast(output_spikes, 'float32')


class AvgPool2DReLU(AveragePooling2D):
    """ batch_size x input_shape x out_shape """
    def __init__(self, pool_size=(2, 2), strides=None, ignore_border=True,
                 label=None, **kwargs):
        self.ignore_border = ignore_border
        super().__init__(pool_size=pool_size, strides=strides)
        if label is not None:
            self.label = label
            self.name = label
        else:
            self.label = self.name

    def get_output(self, train=False):

        # Recurse
        inp, time, updates = get_input(self)

        # CALCULATE SYNAPTIC SUMMED INPUT
        self.impulse = pool.pool_2d(inp, ds=self.pool_size, st=self.strides,
                                    ignore_border=self.ignore_border,
                                    mode='average_inc_pad')

        output_spikes = update_neurons(self, self.impulse, time, updates)
        self.updates = updates
        return T.cast(output_spikes, 'float32')
