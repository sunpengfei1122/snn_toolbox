# -*- coding: utf-8 -*-
"""Keras model parser.

@author: rbodo
"""

import numpy as np
import keras.backend as k
from snntoolbox.parsing.utils import *
from torch.autograd import Variable
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import onnx

class ModelParser(AbstractModelParser):

    def parse(self):
        """Extract the essential information about a neural network.

        This method serves to abstract the conversion process of a network from
        the language the input model was built in (e.g. Keras or Lasagne).

        The methods iterates over all layers of the input model and writes the
        layer specifications and parameters into `_layer_list`. The keys are
        chosen in accordance with Keras layer attributes to facilitate
        instantiation of a new, parsed Keras model (done in a later step by
        `build_parsed_model`).

        This function applies several simplifications and adaptations to
        prepare the model for conversion to spiking. These modifications
        include:

        - Removing layers only used during training (Dropout,
          BatchNormalization, ...)
        - Absorbing the parameters of BatchNormalization layers into the
          parameters of the preceeding layer. This does not affect performance
          because batch-norm-parameters are constant at inference time.
        - Removing ReLU activation layers, because their function is inherent
          to the spike generation mechanism. The information which nonlinearity
          was used in the original model is preserved in the ``activation`` key
          in `_layer_list`. If the output layer employs the softmax function, a
          spiking version is used when testing the SNN in INIsim or MegaSim
          simulators.
        """

        layers = self.get_layer_iterable()
        snn_layers = eval(self.config.get('restrictions', 'snn_layers'))

        name_map = {}
        idx = 0
        inserted_flatten = False
        for layer in layers:
            layer_type = self.get_type(layer)

            # Absorb BatchNormalization layer into parameters of previous layer
            if layer_type == 'BatchNormalization':
                parameters_bn = list(self.get_batchnorm_parameters(layer))
                parameters_bn, axis = parameters_bn[:-1], parameters_bn[-1]
                inbound = self.get_inbound_layers_with_parameters(layer)
                assert len(inbound) == 1, \
                    "Could not find unique layer with parameters " \
                    "preceeding BatchNorm layer."
                prev_layer = inbound[0]
                prev_layer_idx = name_map[str(id(prev_layer))]
                parameters = list(
                    self._layer_list[prev_layer_idx]['parameters'])
                prev_layer_type = self.get_type(prev_layer)
                print("Absorbing batch-normalization parameters into " +
                      "parameters of previous {}.".format(prev_layer_type))

                _depthwise_conv_names = ['DepthwiseConv2D',
                                         'SparseDepthwiseConv2D']
                _sparse_names = ['Sparse', 'SparseConv2D',
                                 'SparseDepthwiseConv2D']
                is_depthwise = prev_layer_type in _depthwise_conv_names
                is_sparse = prev_layer_type in _sparse_names

                if is_sparse:
                    args = [parameters[0], parameters[2]] + parameters_bn
                else:
                    args = parameters[:2] + parameters_bn

                kwargs = {
                    'axis': axis,
                    'image_data_format': keras.backend.image_data_format(),
                    'is_depthwise': is_depthwise}

                params_to_absorb = absorb_bn_parameters(*args, **kwargs)

                if is_sparse:
                    # Need to also save the mask associated with sparse layer.
                    params_to_absorb += (parameters[1],)

                self._layer_list[prev_layer_idx]['parameters'] = \
                    params_to_absorb

            if layer_type == 'GlobalAveragePooling2D':
                print("Replacing GlobalAveragePooling by AveragePooling "
                      "plus Flatten.")
                a = 1 if keras.backend.image_data_format() == 'channels_last' \
                    else 2
                self._layer_list.append(
                    {'layer_type': 'AveragePooling2D',
                     'name': self.get_name(layer, idx, 'AveragePooling2D'),
                     'pool_size': (layer.input_shape[a: a + 2]),
                     'inbound': self.get_inbound_names(layer, name_map),
                     'strides': [1, 1]})
                name_map['AveragePooling2D' + str(idx)] = idx
                idx += 1
                num_str = str(idx) if idx > 9 else '0' + str(idx)
                shape_string = str(np.prod(layer.output_shape[1:]))
                self._layer_list.append(
                    {'name': num_str + 'Flatten_' + shape_string,
                     'layer_type': 'Flatten',
                     'inbound': [self._layer_list[-1]['name']]})
                name_map['Flatten' + str(idx)] = idx
                idx += 1
                inserted_flatten = True

            if layer_type not in snn_layers:
                print("Skipping layer {}.".format(layer_type))
                continue


            print("Parsing layer {}.".format(layer_type))

            if layer_type == 'MaxPooling2D' and \
                    self.config.getboolean('conversion', 'max2avg_pool'):
                print("Replacing max by average pooling.")
                layer_type = 'AveragePooling2D'

            inbound = self.get_inbound_names(layer, name_map)
            attributes = self.initialize_attributes(layer)

            attributes.update({'layer_type': layer_type,
                               'name': self.get_name(layer, idx),
                               'inbound': inbound})

            if layer_type == 'Dense':
                self.parse_dense(layer, attributes)

            if layer_type == 'Sparse':
                self.parse_sparse(layer, attributes)

            if layer_type in {'Conv1D', 'Conv2D'}:
                self.parse_convolution(layer, attributes)

            if layer_type == 'SparseConv2D':
                self.parse_sparse_convolution(layer, attributes)

            if layer_type == 'DepthwiseConv2D':
                self.parse_depthwiseconvolution(layer, attributes)

            if layer_type == 'SparseDepthwiseConv2D':
                self.parse_sparse_depthwiseconvolution(layer, attributes)

            if layer_type in ['Sparse', 'SparseConv2D',
                              'SparseDepthwiseConv2D']:
                weights, bias, mask = attributes['parameters']

                weights, bias = modify_parameter_precision(
                    weights, bias, self.config, attributes)

                attributes['parameters'] = (weights, bias, mask)

                self.absorb_activation(layer, attributes)

            if layer_type in {'Dense', 'Conv1D', 'Conv2D', 'DepthwiseConv2D'}:
                weights, bias = attributes['parameters']
                weights, bias = modify_parameter_precision(
                    weights, bias, self.config, attributes)

                attributes['parameters'] = (weights, bias)
                self.absorb_activation(layer, attributes)

            if 'Pooling' in layer_type:
                self.parse_pooling(layer, attributes)

            if layer_type == 'Concatenate':
                self.parse_concatenate(layer, attributes)

            self._layer_list.append(attributes)

            # Map layer index to layer id. Needed for inception modules.
            name_map[str(id(layer))] = idx

            idx += 1
        print('')
        
    def get_layer_iterable(self):
        return self.input_model.layers

    def get_type(self, layer):
        from snntoolbox.parsing.utils import get_type
        return get_type(layer)

    def get_batchnorm_parameters(self, layer):
        mean = k.get_value(layer.moving_mean)
        var = k.get_value(layer.moving_variance)
        var_eps_sqrt_inv = 1 / np.sqrt(var + layer.epsilon)
        gamma = np.ones_like(mean) if layer.gamma is None else \
            k.get_value(layer.gamma)
        beta = np.zeros_like(mean) if layer.beta is None else \
            k.get_value(layer.beta)
        axis = layer.axis

        return [mean, var_eps_sqrt_inv, gamma, beta, axis]

    def get_inbound_layers(self, layer):
        from snntoolbox.parsing.utils import get_inbound_layers
        return get_inbound_layers(layer)

    @property
    def layers_to_skip(self):
        # noinspection PyArgumentList
        return AbstractModelParser.layers_to_skip.fget(self)

    def has_weights(self, layer):
        from snntoolbox.parsing.utils import has_weights
        return has_weights(layer)

    def initialize_attributes(self, layer=None):
        attributes = AbstractModelParser.initialize_attributes(self)
        attributes.update(layer.get_config())
        return attributes

    def get_input_shape(self):
        return tuple(self.get_layer_iterable()[0].batch_input_shape[1:])

    def get_output_shape(self, layer):
        return layer.output_shape

    def parse_sparse(self, layer, attributes):
        return self.parse_dense(layer, attributes)
    
    

    def parse_dense(self, layer, attributes):
        attributes['parameters'] = list(layer.get_weights())
        if layer.bias is None:
            attributes['parameters'].insert(
                1, np.zeros(layer.output_shape[1]))
            attributes['parameters'] = tuple(attributes['parameters'])
            attributes['use_bias'] = True

    def parse_sparse_convolution(self, layer, attributes):
        return self.parse_convolution(layer, attributes)

    def parse_convolution(self, layer, attributes):
        attributes['parameters'] = list(layer.get_weights())
        if layer.bias is None:
            attributes['parameters'].insert(1, np.zeros(layer.filters))
            attributes['parameters'] = tuple(attributes['parameters'])
            attributes['use_bias'] = True
        assert layer.data_format == k.image_data_format(), (
            "The input model was setup with image data format '{}', but your "
            "keras config file expects '{}'.".format(layer.data_format,
                                                     k.image_data_format()))

    def parse_sparse_depthwiseconvolution(self, layer, attributes):
        return self.parse_depthwiseconvolution(layer, attributes)

    def parse_depthwiseconvolution(self, layer, attributes):
        attributes['parameters'] = list(layer.get_weights())
        if layer.bias is None:
            a = 1 if layer.data_format == 'channels_first' else -1
            attributes['parameters'].insert(1, np.zeros(
                layer.depth_multiplier * layer.input_shape[a]))
            attributes['parameters'] = tuple(attributes['parameters'])
            attributes['use_bias'] = True

    def parse_pooling(self, layer, attributes):
        pass

    def get_activation(self, layer):

        return layer.activation.__name__

    def get_outbound_layers(self, layer):

        from snntoolbox.parsing.utils import get_outbound_layers

        return get_outbound_layers(layer)

    def parse_concatenate(self, layer, attributes):
        pass

   
def load(path, filename, **kwargs):
    """Load network from file.

    Parameters
    ----------

    path: str
        Path to directory where to load Pytorch model parameters from.

    filename: str
        Name of file to load Pytorch model from.
        
    path_model: str
        Path to directory where to load Pytorch model from

    Returns
    -------

    : dict[str, Union[keras.models.Sequential, function]]
        A dictionary of objects that constitute the input model. It must
        contain the following two keys:

        - 'model': keras.models.Sequential
            Keras model instance of the network.
        - 'val_fn': function
            Function that allows evaluating the original model.
    """

    import os
    from keras import models, metrics

    filepath = str(os.path.join(path, filename))
    
    #Create dummy variable with correct shape 
    dummy_input = np.random.uniform(0, 1, (1, 1, 28, 28))  
    dummy_input = Variable(torch.FloatTensor(dummy_input))
    input_shapes = [(1, 28, 28)]
    
    #Use dummy-variable to trace the Pytorch model
    from snntoolbox.pytorch2keras.onnx2keras.converter import onnx_to_keras

    
    #load and trace the Pytorch model
    import sys 
    path_model = os.path.join(path, "../")
    sys.path.append(path_model)
    import my_models
    from my_models import my_model 
    model = my_model
    
    
    #Recommended save method
    model.load_state_dict(torch.load(filepath + '.pth'))
    dummy_output = model(dummy_input)
    
    if isinstance(dummy_output, torch.autograd.Variable):
        dummy_output = [dummy_output]
    if not isinstance(dummy_input, list):
        dummy_input = [dummy_input]
    dummy_input = tuple(dummy_input)
    #export as onnx model, and then reload
    input_names = ['input_{0}'.format(i) for i in range(len(dummy_input))]
    output_names = ['output_{0}'.format(i) for i in range(len(dummy_output))]
    print("output_names", output_names)
    torch.onnx.export(model, dummy_input, 'model.onnx', input_names=input_names, output_names=output_names)
    onnx_model = onnx.load('model.onnx')
    k_model = onnx_to_keras(onnx_model=onnx_model, input_names=input_names,input_shapes=input_shapes)
    # save the keras model
    keras.models.save_model(k_model, os.path.join(filepath + '.h5'))
    

    from snntoolbox.parsing.utils import get_custom_activations_dict, \
            assemble_custom_dict, get_custom_layers_dict
    filepath_custom_objects = kwargs.get('filepath_custom_objects', None)
    if filepath_custom_objects is not None:
        filepath_custom_objects = str(filepath_custom_objects)  # python 2

    custom_dicts = assemble_custom_dict(
        get_custom_activations_dict(filepath_custom_objects),
        get_custom_layers_dict())
    try:
        model = models.load_model(filepath + '.h5', custom_dicts)
    except OSError as e:
        print(e)
        print("Trying to load without '.h5' extension.")
        model = models.load_model(filepath, custom_dicts)
        #model.compile(model.optimizer, model.loss,
        #              ['accuracy', metrics.top_k_categorical_accuracy])
    model.compile('sgd', 'categorical_crossentropy',
                      ['accuracy', metrics.top_k_categorical_accuracy])
    model.summary()
    return {'model': model, 'val_fn': model.evaluate}


def evaluate(val_fn, batch_size, num_to_test, x_test=None, y_test=None,
             dataflow=None):
    """Evaluate the original ANN.

    Can use either numpy arrays ``x_test, y_test`` containing the test samples,
    or generate them with a dataflow
    (``Keras.ImageDataGenerator.flow_from_directory`` object).

    Parameters
    ----------

    val_fn:
        Function to evaluate model.

    batch_size: int
        Batch size

    num_to_test: int
        Number of samples to test

    x_test: Optional[np.ndarray]

    y_test: Optional[np.ndarray]

    dataflow: keras.ImageDataGenerator.flow_from_directory
    """

    if x_test is not None:
        score = val_fn(x_test, y_test, batch_size, verbose=0)
    else:
        score = np.zeros(3)
        batches = int(num_to_test / batch_size)
        for i in range(batches):
            x_batch, y_batch = dataflow.next()
            score += val_fn(x_batch, y_batch, batch_size, verbose=0)
        score /= batches

    print("Top-1 accuracy: {:.2%}".format(score[1]))
    print("Top-5 accuracy: {:.2%}\n".format(score[2]))

    return score[1]
