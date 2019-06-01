import keras
import keras.layers as KL
import keras.models as KM
from keras.callbacks import ModelCheckpoint
import os

MODEL_NAME = 'bi_rnn_test'
MODEL_SAVE_PATH = './models/checkpoints/{}/'.format(MODEL_NAME)
CHECKPOINT_PATH = MODEL_SAVE_PATH + '{epoch:02d}-{val_acc:.4f}.hdf5'

TIMESTEPS = 256
DIM_MELODY = 90
DIM_LATENT = 512

# Attept to make the MODEL_SAVE_PATH directory if it doesn't already exist
try:
    os.mkdir(MODEL_SAVE_PATH)
except FileExistsError:
    pass

def get_model():
    
    checkpoint = ModelCheckpoint(CHECKPOINT_PATH, monitor='val_acc',
                                 verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    input_layer = KL.Input(shape=(TIMESTEPS, DIM_MELODY), name='input')
    layer = KL.Bidirectional(
        KL.CuDNNLSTM(128, return_sequences=True, name='bi_lstm_1'))(input_layer)
    layer = KL.Bidirectional(KL.CuDNNLSTM(128, name='bi_lstm_2'))(layer)
    output_layer = KL.Dense(DIM_LATENT, activation='linear', name='output')(layer)
    return KM.Model(inputs=input_layer, outputs=output_layer)