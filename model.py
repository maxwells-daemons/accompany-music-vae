'''
Defines the accompaniment model.

This is the "surrogate encoder" model which predicts the latent code inferred
by MusicVAE's trio encoder from just the melody. The model is a multi-layer
bidirectional LSTM RNN.

Functions
---------
get_model() : keras Model
    Gets the compiled surrogate encoder model.
'''


import os

import keras
import keras.layers as KL
import keras.models as KM

from constants import (TIMESTEPS, DIM_MELODY, DIM_LATENT)


def get_model(name='surr_encoder', n_units=128, n_layers=2, optimizer=None,
              initial_weights=None):
    '''
    Get the compiled surrogate encoder model.

    Parameters
    ----------
    name : str (optional)
        The model name.
    n_units : int (optional)
        How many RNN cells to use per layer.
    n_layers : int (optional)
        How many bidirectional LSTM layers to use.
    optimizer : None or keras Optimizer (optional)
        The optimizer to use.
        If None, uses a default rmsprop optimizer.
    initial_weights : None or str path to hdf5  (optional)
        Path to weights used to initialize the model.
        If None, initialize randomly (default).

    Returns
    -------
    keras Model
        The compiled model.

    Raises
    ------
    AssertionError
        If initial_weights is not None but is not a valid path.
    '''
    input_layer = KL.Input(shape=(TIMESTEPS, DIM_MELODY), name='input')
    layer = input_layer
    for i in range(n_layers):
        layer = KL.Bidirectional(
            KL.CuDNNLSTM(n_units,
                         return_sequences=(i != n_layers - 1),
                         name='bi_lstm_{}'.format(i))
        )(layer)
    output_layer = KL.Dense(DIM_LATENT,
                            activation='linear', name='output')(layer)
    model = KM.Model(inputs=input_layer, outputs=output_layer)

    if initial_weights:
        assert(os.path.exists(initial_weights))
        model.load_weights(initial_weights)

    optimizer = optimizer or keras.optimizers.rmsprop(lr=3e-4, clipnorm=1.)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.name = name

    return model
