'''
Code for training a surrogate encoder model.

Functions
---------
train_model(model, batch_size, epochs, data_path)
    Train a surrogate encoder model.

Notes
-----
train_model() is also a Click script that can be run with
`python train.py [options]`.
'''

import os

import click
import h5py
import keras

from constants import (CHECKPOINT_PATH, LOG_PATH)
from hdf5_sequence import HDF5Sequence
from model import get_model


@click.command()
@click.option('--model', default=None)
@click.option('--batch_size', type=int, default=16)
@click.option('--epochs', type=int, default=10)
@click.option('--data_path', type=click.Path(exists=True),
              default='./data/lmd_full/lmd_full_split.h5')
def train_model(model, batch_size, epochs, data_path):
    '''
    Train a surrogate encoder model.

    Parameters
    ----------
    model : None, keras Model, or str path to hdf5
        The model to train.
        If None, initializes a new default model.
        If an existing model, uses it as-is.
        If a path, loads the model from the path.
    batch_size : int
        Batch size for training.
    epochs : int
        Number of epochs to train for.
    data_path : str path to hdf5
        Path to the file of training data.

    Raises
    ------
    AssertionError
        If model is a str but not a valid path.
    '''

    if not model:
        model = get_model()
    elif isinstance(model, str):
        assert(os.path.exists(model))
        model = keras.load_model(model)
    # Otherwise, assume the model is a Keras model

    # TODO: LambdaCallback to produce and save samples at each epoch
    checkpointer = keras.callbacks.ModelCheckpoint(
        os.path.join(
            CHECKPOINT_PATH,
            model.name + '_train_{epoch:02d}-{val_loss:.4f}.hdf5'
        ),
        save_best_only=False, verbose=1
    )
    logger = keras.callbacks.CSVLogger(
        os.path.join(LOG_PATH, '{}_train.log'.format(model.name)),
        append=True
    )
    callbacks = [checkpointer, logger]

    data_file = h5py.File(data_path, 'r')
    data_dir = os.path.dirname(data_path)
    train_seq = HDF5Sequence(
        data_file, batch_size,
        index_path=os.path.join(data_dir, 'train_indices.csv'))
    val_seq = HDF5Sequence(
        data_file, batch_size,
        index_path=os.path.join(data_dir, 'val_indices.csv'))

    model.fit_generator(train_seq, steps_per_epoch=len(train_seq),
                        validation_data=val_seq, validation_steps=len(val_seq),
                        max_queue_size=128, workers=32, epochs=20,
                        callbacks=callbacks)


if __name__ == '__main__':
    train_model()
