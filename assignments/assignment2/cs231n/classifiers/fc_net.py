from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # \
        lst_dim = input_dim
        for i in range(1,self.num_layers):
            # print(f"W%d,b%d = [%d,%d]"%(i,i,lst_dim,hidden_dims[i-1]))
            self.params[f'W{i}'] = np.random.normal(loc = 0,scale = weight_scale,size = (lst_dim,hidden_dims[i-1]))
            self.params[f'b{i}'] = np.zeros((hidden_dims[i-1],))
            if self.normalization == "batchnorm":
               self.params[f'gamma{i}'] = np.ones((hidden_dims[i-1],))
               self.params[f'beta{i}'] = np.zeros((hidden_dims[i-1],))
            lst_dim = hidden_dims[i-1]
        
        # print(f"W%d,b%d = [%d,%d]"%(maxx+1,maxx+1,lst_dim,num_classes))
        self.params[f'W%d'%self.num_layers] = np.random.normal(loc = 0,scale = weight_scale,size = (lst_dim,num_classes))
        self.params[f'b%d'%self.num_layers] = np.zeros((num_classes,))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Now I just write code without batch/layer norm and dropout layer--2019/8/18
        # Today I add dropout layer after ReLU--2019/8/19
        # Today I add batch normalization layer before ReLU--2019/8/20
        XX = X.reshape(X.shape[0], -1).copy()
        N, D = XX.shape

        # the params between input layer and first hidden layer
        hidden_in_value = [] # It seems no need
        hidden_mid_value = []
        hidden_out_value = []

        cache_af_value = []
        cache_relu_value = []
        cache_bn_value = []
        cache_ln_value = []
        cache_dropout_value = []

        hidden_out = X
        # the forward pass between input layer/hidden layer and hidden layer
        num_hidden_layers = self.num_layers - 1
        for i in range(num_hidden_layers):
            str_W = 'W' + str('%d' %(i+1))
            str_b = 'b' + str('%d' %(i+1))
            str_gamma = 'gamma' + str('%d' %(i+1))
            str_beta = 'beta' + str('%d' %(i+1))

            hidden_in, cache_af = affine_forward(hidden_out, self.params[str_W], self.params[str_b])
            hidden_in_value.append(hidden_in) # recorde forward pass value for backprop,not used later
            cache_af_value.append(cache_af)
            if self.normalization is not None:
                if self.normalization == 'batchnorm':
                    hidden_norm, cache_bn = batchnorm_forward(hidden_in, self.params[str_gamma],
                    self.params[str_beta], self.bn_params[i]) # do not forget gamma & beta
                    cache_bn_value.append(cache_bn)
                    
                if self.normalization == 'layernorm':
                    hidden_norm, cache_ln = layernorm_forward(hidden_in, self.params[str_gamma], 
                    self.params[str_beta], self.bn_params[i])
                    cache_ln_value.append(cache_ln)
            else:
                hidden_norm = hidden_in

            hidden_mid, cache_relu = relu_forward(hidden_norm)
            hidden_mid_value.append(hidden_mid)
            cache_relu_value.append(cache_relu)

            if self.use_dropout:
                hidden_out, cache_dropout = dropout_forward(hidden_mid, self.dropout_param)
                cache_dropout_value.append(cache_dropout)
            else:
                hidden_out = hidden_mid
            hidden_out_value.append(hidden_out) # recorde forward pass value for backprop

        # the params between last hidden layer and output layer, no gamma and beta
        str_W = 'W' + str('%d' %(num_hidden_layers + 1))
        str_b = 'b' + str('%d' %(num_hidden_layers + 1))
        scores = hidden_out.dot(self.params[str_W]) + self.params[str_b]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Now I just write code without batch/layer norm and dropout layer--2019/8/18
        ############################ compute loss ##################################
        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0.0
        for i in range(self.num_layers):
            str_W = 'W' + str('%d' %(i+1))
            reg_loss += np.sum(self.params[str_W] ** 2)

        reg_loss = 0.5 * self.reg * reg_loss
        loss = data_loss + reg_loss

        ############################ compute grads #################################
        str_W = 'W' + str('%d' % self.num_layers)
        str_b = 'b' + str('%d' % self.num_layers)
        grads[str_W] = np.dot(hidden_out.T, dscores) + self.reg * self.params[str_W]
        grads[str_b] = np.sum(dscores, axis=0)

        grad_x_in = np.dot(dscores, self.params[str_W].T)

        for i in range(num_hidden_layers, 0, -1):

            if self.use_dropout:
                grad_x_mid = dropout_backward(grad_x_in, cache_dropout_value[i-1])
            else:
                grad_x_mid = grad_x_in

            # Pass ReLU
            grad_x_norm = relu_backward(grad_x_mid, cache_relu_value[i-1])

            str_gamma = 'gamma' + str('%d' % i)
            str_beta = 'beta' + str('%d' % i)
            if self.normalization is not None:
                if self.normalization == 'batchnorm':
                    grad_x_out, grads[str_gamma], grads[str_beta] = \
                        batchnorm_backward_alt(grad_x_norm, cache_bn_value[i-1]) # Do not forget dgamma, dbeta
                if self.normalization == 'layernorm':
                    grad_x_out, grads[str_gamma], grads[str_beta] = \
                        layernorm_backward(grad_x_norm, cache_ln_value[i-1])
            else:
                grad_x_out = grad_x_norm

            str_W = 'W' + str('%d' % i)
            str_b = 'b' + str('%d' % i)

            grad_x_in, grads[str_W], grads[str_b] = affine_backward(grad_x_out, cache_af_value[i-1])
            grads[str_W] += self.reg * self.params[str_W]


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads