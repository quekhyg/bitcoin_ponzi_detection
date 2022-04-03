#Standard Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import metrics
from tensorflow.keras.metrics import Metric

#Custom Imports
import ml_functions as ml

class NeuralNetwork:
    n_NNs = 0
    
    def __init__(self, x_train, y_train, x_test, y_test,
                 layer_types, n_nodes, activations, dropout_param, optimiser, loss_fn, learn_rate, filepath, long_folder = True):
        #layer_types is a sequential list of layer types, with each item being 'dense', 'dropout' or 'batch_norm'
        #n_nodes is a sequential list of integers, with each integer indicating the number of nodes for its respective layer
        #activations is a sequential list of activation functions, with each function to be applied to its respective layer
        #dropout_param is a fraction between 0 and 1 which indicates the probability of dropping any particular node
        
        #optimiser is an instance of tf.keras.optimizers
        #loss_fn is a string name of loss function or an instance of tf.keras.losses.Loss
        #learn_rate is a small positive number used for gradient descent, as an argument for optimiser
        
        NeuralNetwork.n_NNs += 1
        self.index = NeuralNetwork.n_NNs
        self.n_fit = 0
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
        self.layer_types = layer_types
        self.n_nodes = n_nodes
        self.activations = activations
        self.dropout_param = dropout_param
        self.optimiser = optimiser
        self.loss_fn = loss_fn
        self.learn_rate = learn_rate
        
        #Creating NN
        self.m = tf.keras.Sequential()
        
        #Hidden layers
        for lay, n_node, act in zip(layer_types, n_nodes, activations):
            if lay == 'dense':
                self.m.add(layers.Dense(n_node, activation = act, dtype = 'float64'))
            elif lay == 'dropout':
                self.m.add(layers.Dropout(dropout_param, dtype = 'float64'))
            elif lay == 'batch_norm':
                self.m.add(layers.BatchNormalization(dtype = 'float64'))
        
        #Output layer
        self.m.add(layers.Dense(1, activation = tf.math.sigmoid, dtype='float64'))
        
        #Compiling
        self.m.compile(optimizer = optimiser(learning_rate = learn_rate), loss = loss_fn,
                       metrics = [metrics.BinaryAccuracy(), metrics.Precision(), metrics.Recall(),
                                  StatefulBinaryFBeta(name = 'F1', beta = 1), StatefulBinaryFBeta(name = 'F2', beta = 2)])
        #StatefulBinaryFBeta(name = 'F1', beta = 1), StatefulBinaryFBeta(name = 'F2', beta = 2)
        
        #Save filepath convention
        self.filepath = filepath
        folder = ''
        if long_folder:
            for lay, n_node, act in zip(self.layer_types, self.n_nodes, self.activations):
                if lay == 'dense':
                    folder += str(int(n_node))
                    if act == tf.nn.relu:
                        folder += 'r_'
                    elif act == tf.nn.leaky_relu:
                        folder += 'lr_'
                    else:
                        folder += 'uk_'
                        print('Error: please define filename convention for new activation function')
                elif lay == 'dropout':
                    folder += 'do'+str(int(self.dropout_param*10))+'_'
                elif lay == 'batch_norm':
                    folder += 'bn_'
            folder += self.optimiser.__name__
            if loss_fn == 'BinaryCrossentropy':
                folder += 'bce_'
            elif loss_fn == 'BinaryFocalCrossentropy':
                folder += 'bfce_'
            else:
                folder += 'uk_'
                print('Error: please define filename convention for new loss function')
            folder += 'f{}'.format(self.index)
        #kwargs are: overwrite = True, include_optimizer = True, save_format = None, signatures = None, options = None, save_traces = True
        else:
            folder = 'f{}'.format(self.index)
        self.folder = folder
    
    def model_fit(self, batch_size, n_epochs, callback_list = [], save_checkpoints = True, validation_split = None):
        #batch_size is the number of data points per batch in training
        #n_epoch is the number of epochs to train for
        if save_checkpoints:
            callback_mcp = tf.keras.callbacks.ModelCheckpoint(
                filepath = os.path.join(self.filepath, self.folder, 'ckpt', 'weights.{epoch:02d}-{val_loss:.2f}'),
                save_freq = 'epoch', save_weights_only = True, verbose = 1,
                monitor = 'val_loss', mode = 'min', save_best_only = True)
            callback_list.append(callback_mcp)
        
        self.n_epochs = n_epochs
        self.curr_epoch = 0
        self.batch_size = batch_size
        if validation_split is None:
            self.result = self.m.fit(self.x_train, self.y_train,
                                     batch_size = batch_size,
                                     epochs = n_epochs,
                                     validation_data = (self.x_test, self.y_test),
                                     shuffle = True,
                                     callbacks = callback_list)
        else:
            self.result = self.m.fit(self.x_train, self.y_train,
                                     batch_size = batch_size,
                                     epochs = n_epochs,
                                     validation_split = validation_split,
                                     shuffle = True,
                                     callbacks = callback_list)
        self.n_fit += 1
    
    def plot_metrics(self, start_epoch, end_epoch):
        end_epoch = min(end_epoch, self.n_epochs)
        n_metrics = len(self.result.history) // 2
        fig, axs = plt.subplots((n_metrics+1)//2, 2, figsize = (12, n_metrics*2))
        fig.suptitle('Evaluation Metrics')
        for i, (k, v) in enumerate(self.result.history.items()):
            if i < n_metrics:
                axs[i//2, i%2].plot(v[start_epoch:end_epoch], label = 'train')
                axs[i//2, i%2].set_xlabel('Epoch Number')
                axs[i//2, i%2].set_xticks(np.arange(0, end_epoch-1, step = 50))
                axs[i//2, i%2].set_ylabel(k)
            else:
                axs[(i-n_metrics-1)//2, i%2].plot(v[start_epoch:end_epoch], label = 'validation')
                axs[(i-n_metrics-1)//2, i%2].legend()
        plt.show()

    def save_model(self, **kwargs):
        try:
            self.m.save(os.path.join(self.filepath, self.folder), **kwargs)
        except:
            self.m.save(os.path.join(self.filepath,'nn{}run{}'.format(self.index, self.n_fit)), **kwargs)
        np.save(os.path.join(self.filepath, self.folder, 'history'), self.result.history)

    def reset_class_index(self):
        Neural_Network.n_NNs = 0
        
class StatefulBinaryFBeta(Metric):   
    def __init__(self, name = 'stateful_binary_fbeta', beta = 1, threshold = 0.5, epsilon = 1e-7, **kwargs):
        # initializing an object of the super class
        super(StatefulBinaryFBeta, self).__init__(name = name, **kwargs)

        # initializing state variables
        self.tp = self.add_weight(name = 'tp', initializer = 'zeros') # initializing true positives 
        self.actual_positive = self.add_weight(name = 'fp', initializer = 'zeros') # initializing actual positives
        self.predicted_positive = self.add_weight(name = 'fn', initializer = 'zeros') # initializing predicted positives

        self.beta = beta
        self.threshold = threshold
        self.epsilon = epsilon

    def update_state(self, ytrue, ypred, sample_weight=None):
        # casting ytrue and ypred as float dtype
        ytrue = tf.cast(ytrue, tf.float32)
        ypred = tf.cast(ypred, tf.float32)

        # setting values of ypred greater than the set threshold to 1 while those lesser to 0
        ypred = tf.cast(tf.greater_equal(ypred, tf.constant(self.threshold)), tf.float32)
        
        self.tp.assign_add(tf.reduce_sum(ytrue*ypred)) # updating true positives attribute
        self.predicted_positive.assign_add(tf.reduce_sum(ypred)) # updating predicted positive attribute
        self.actual_positive.assign_add(tf.reduce_sum(ytrue)) # updating actual positive attribute

    def result(self):
        self.precision = self.tp/(self.predicted_positive+self.epsilon) # calculates precision
        self.recall = self.tp/(self.actual_positive+self.epsilon) # calculates recall

        # calculating fbeta
        self.fb = (1+self.beta**2)*self.precision*self.recall / (self.beta**2*self.precision + self.recall + self.epsilon)
    
        return self.fb

    def reset_states(self):
        self.tp.assign(0) # resets true positives to zero
        self.predicted_positive.assign(0) # resets predicted positives to zero
        self.actual_positive.assign(0) # resets actual positives to zero