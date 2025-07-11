from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p)
        loss -= logp[y[i]]  # negative log probability is the loss
        for j in range(W.shape[0]):
            for k in range(num_classes):
                if k == y[i]:
                    dW[j,k] += (p[k] - 1) * X[i,j]
                else:
                    dW[j,k] += p[k] * X[i,j]
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train + 2 * reg * W # 注意这边是求导完的

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    scores = X.dot(W)
    scores -= np.max(scores,axis = 1,keepdims=True)
    dp = np.exp(scores)
    prob = dp / np.sum(dp,axis = 1,keepdims = True)
    
    cp = -np.log(prob[np.arange(num_train),y])
    loss = np.sum(cp) / num_train + reg * np.sum(W * W)

    ds = prob
    ds[np.arange(num_train),y] -= 1
    ds /= num_train

    dW = np.dot(X.T,ds) + 2 * reg * W

    # num_classes = W.shape[1]
    # num_train = X.shape[0]
    # for i in range(num_train):
    #     scores = X[i].dot(W)

    #     # compute the probabilities in numerically stable way
    #     scores -= np.max(scores)
    #     p = np.exp(scores)
    #     p /= p.sum()  # normalize
    #     logp = np.log(p)
    #     loss -= logp[y[i]]  # negative log probability is the loss
    #     for j in range(W.shape[0]):
    #         for k in range(num_classes):
    #             if k == y[i]:
    #                 dW[j,k] += (p[k] - 1) * X[i,j]
    #             else:
    #                 dW[j,k] += p[k] * X[i,j]
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################

    # loss = loss / num_train + reg * np.sum(W * W)
    # dW = dW / num_train + 2 * reg * W # 注意这边是求导完的

    return loss, dW
