import sys, os
sys.path.append(os.pardir)
from collections import OrderedDict
from shared_code.layer import *
from shared_code.gradient import numerical_gradient

class MultiLayerNetExtend:
    """Multi-layer neural network with extended version of fully coupled

        Weiht Decay、Dropout、Batch Normalizationの機能を持つ

    Parameters
    ----------
    input_size : 入力サイズ（MNISTの場合は784）
    hidden_size_list : 隠れ層のニューロンの数のリスト（e.g. [100, 100, 100]）
    output_size : 出力サイズ（MNISTの場合は10）
    activation : 'relu' or 'sigmoid'
    weight_init_std : 重みの標準偏差を指定（e.g. 0.01）
        'relu'または'he'を指定した場合は「Heの初期値」を設定
        'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
    weight_decay_lambda : Weight Decay（L2ノルム）の強さ
    use_dropout: Dropoutを使用するかどうか
    dropout_ration : Dropoutの割り合い
    use_batchNorm: Batch Normalizationを使用するかどうか
    """

    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0,
                 use_dropout=False, dropout_ration=0.5, use_batchnorm=False):
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.use_dropout = use_dropout
        self.weight_decay_lambda = weight_decay_lambda
        self.use_batchnorm = use_batchnorm
        self.params = {}

        # initialize of weight
        self.__init_weight(weight_init_std)

        # Make a layer
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            if self.use_batchnorm:
                self.params[f'gamma{str(idx)}'] = np.ones(hidden_size_list[idx-1])
                self.params[f'beta{str(idx)}'] = np.zeros(hidden_size_list[idx-1])
                self.layers[f'BatchNorm{str(idx)}'] = BatchNormalization(self.params[f'gamma{str(idx)}'], self.params[f'beta{idx}'])

            self.layers[f'Activation_function{idx}'] = activation_layer[activation]()

            if self.use_dropout:
                self.layers[f'Dropout{idx}'] = Dropout(dropout_ration)

        idx = self.hidden_layer_num + 1
        self.layers[f'Affine{idx}'] = Affine(self.params[f'W{idx}'], self.params[f'b{idx}'])

        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """ Settings the initial value of weight
        
        Parameters
        ----------
        weight_init_std: Specify the standard deviation of the weights
            selected: 'relu', 'he' -> "Initial value of He"
            selected: 'sigmoid', 'xavier' -> "Initial value of Xavier"
        """

        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]

        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1]) # recommended a initialize value if use case relu
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1]) # reccommended a initialize value if use case sigmoid
            self.params[f'W{idx}'] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params[f'b{idx}'] = np.zeros(all_size_list[idx])

        # debug
        # print(f"self.params: {self.params.keys()}")

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)

        return x

    def loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params[f'W{idx}']
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x, train_flg=False)
        y = np.argmax(y, axis=1)

        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

    def numerical_gradient(self, x, t):

        loss_W = lambda W: self.loss(x, t, train_flg=True)

        grads = {}

        for idx in range(1, self.hidden_layer_num + 2):
            grads[f'W{idx}'] = numerical_gradient(loss_W, self.params[f'W{idx}'])
            grads[f'b{idx}'] = numerical_gradient(loss_W, self.params[f'b{idx}'])

            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads[f'gamma{idx}'] = numerical_gradient(loss_W, self.params[f'gamma{idx}'])
                grads[f'beta{idx}'] = numerical_gradient(loss_W, self.params[f'beta{idx}'])

        return grads

    def gradient(self, x, t):

        # forward
        self.loss(x, t, train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())

        # debug
        # print(f"layers: {layers}")
        # print(f"layers.reverse: {layers.reverse()}")

        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        # Settings
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads[f'W{idx}'] = self.layers[f'Affine{idx}'].dW + self.weight_decay_lambda * self.params[f'W{idx}']
            grads[f'b{idx}'] = self.layers[f'Affine{idx}'].db

            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads[f'gamma{idx}'] = self.layers[f'BatchNorm{idx}'].dgamma
                grads[f'beta{idx}'] = self.layers[f'BatchNorm{idx}'].dbeta

        return grads