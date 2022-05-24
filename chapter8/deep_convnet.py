import pickle
import numpy as np
from collections import OrderedDict
from shared_code.layer import *
from functions.index_relu import index_relu
import sys
import os
sys.path.append(os.pardir)


class DeepConvNet:
    """Accuracy 99% ConvNet

    Network configuration
    conv - relu - conv - relu - pool -
    conv - relu - conv - relu - pool - 
    conv - relu - conv - relu - pool -
    affine - relu - dropout - affine - dropout - softmax
    """

    def __init__(self, input_dim=(1, 28, 28),
                 conv_param_1={'filter_num': 16,
                               'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_2={'filter_num': 16,
                               'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_3={'filter_num': 32,
                               'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_4={'filter_num': 32,
                               'filter_size': 3, 'pad': 2, 'stride': 1},
                 conv_param_5={'filter_num': 64,
                               'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_6={'filter_num': 64,
                               'filter_size': 3, 'pad': 1, 'stride': 1},
                 hidden_size=50, output_size=10):

        # initialize of weight
        # How many connections do each neuron in each layer have with the neurons in the presheaf
        pre_node_nums = np.array(
            [1*3*3, 16*3*3, 16*3*3, 32*3*3, 32*3*3, 64*3*3, 64*4*4, hidden_size])
        weight_init_scales = np.sqrt(2.0 / pre_node_nums)

        self.params = {}
        pre_channel_num = input_dim[0]

        for idx, conv_param in enumerate([conv_param_1, conv_param_2, conv_param_3, conv_param_4, conv_param_5, conv_param_6]):
            self.params[f'W{idx+1}'] = weight_init_scales[idx] * np.random.randn(
                conv_param['filter_num'], pre_channel_num, conv_param['filter_size'], conv_param['filter_size'])
            self.params[f'b{idx+1}'] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']
        self.params['W7'] = weight_init_scales[6] * \
            np.random.randn(64*5*5, hidden_size)
        self.params['b7'] = np.zeros(hidden_size)
        self.params['W8'] = weight_init_scales[7] * \
            np.random.randn(hidden_size, output_size)
        self.params['b8'] = np.zeros(output_size)

        # Create layer
        self.layers = []

        for i in range(1, 7):
            self.layers.append(Convolution(self.params[f'W{i}'], self.params[f'b{i}'],
                                            conv_param_1['stride'], conv_param_2['pad']))
            if i % 2 == 0:
                self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        
        # relu layer
        for i in index_relu():
            if i == 9: 
                continue
            else:
                self.layers.append(Relu())