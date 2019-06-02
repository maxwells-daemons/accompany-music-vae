import keras
import keras.layers as KL
import keras.models as KM
import os

from constants import *

# Attept to make the MODEL_SAVE_PATH directory if it doesn't already exist
try:
    os.mkdir(MODEL_SAVE_PATH)
except FileExistsError:
    pass

def get_model():
    input_layer = KL.Input(shape=(TIMESTEPS, DIM_MELODY), name='input')
    layer = KL.Bidirectional(
        KL.CuDNNLSTM(128, return_sequences=True, name='bi_lstm_1'))(input_layer)
    layer = KL.Bidirectional(KL.CuDNNLSTM(128, name='bi_lstm_2'))(layer)
    output_layer = KL.Dense(DIM_LATENT, activation='linear', name='output')(layer)
    return KM.Model(inputs=input_layer, outputs=output_layer)