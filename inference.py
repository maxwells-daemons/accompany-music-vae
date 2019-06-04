'''
Code to generate accompaniments.

Uses a trained surrogate encoder and a pretrained decoder to perform inference.

Functions
---------
generate_accompaniments() : [NoteSequence]
    Generate accompaniment for a list of sequences.

Examples
--------
```
model = keras.models.load_model(...)
inputs = [
    './data/lmd_clean/raw/Michael Jackson/Beat It.mid',
    './data/lmd_clean/raw/Black Sabbath/Iron Man.mid'
]
accompaniments = generate_accompaniments(inputs, model)
```
'''


import os
from copy import deepcopy

import numpy as np
import magenta.music as mm
from magenta.models.music_vae import configs, trained_model

from constants import (MUSICVAE_MODEL_NAME, MUSICVAE_MODEL_PATH)
from utils import (strip_to_melody, remove_melody)


def generate_accompaniments(sequences, surrogate_encoder, musicvae=None,
                            stitch=True, extract_melody=False,
                            remove_controls=False, temperature=0.1):
    '''
    Generate accompaniment for a collection of input sequences.

    Parameters
    ----------
    sequences : [str path to midi or NoteSequence]
        The input sequences.
    surrogate_encoder : keras Model
        The model to map melodies to latent vectors.
    musicvae : None or Magenta MusicVAE (optional)
        The MusicVAE to use for decoding.
        If None, loads the default MusicVAE.
        NOTE: For quickly performing inference on multiple input batches,
        preload the default MusicVAE outside this function and pass it in.
    stitch : bool (optional)
        Whether to stitch in the original melody or leave the decoded sequence.
    extract_melody : bool (optional)
        Whether to treat the input as a trio and extract the melody.
    remove_controls : bool (optional)
        Whether to delete tempo changes, time changes, etc from the base midi.
    temperature : float (optional)
        Temperature to use in the trio decoder.

    Returns
    -------
    [NoteSequence]
        The input sequences along with generated accompaniment.
    '''

    config = configs.CONFIG_MAP[MUSICVAE_MODEL_NAME]

    musicvae = musicvae or trained_model.TrainedModel(
        config, batch_size=16,
        checkpoint_dir_or_path=os.path.join(MUSICVAE_MODEL_PATH,
                                            MUSICVAE_MODEL_NAME + '.ckpt')
    )

    melodies = []

    for seq_ in sequences:
        seq = deepcopy(seq_)

        if isinstance(seq, str):
            seq = mm.midi_to_sequence_proto(seq)

        if remove_controls:
            del seq.tempos[:]
            del seq.time_signatures[:]
            del seq.control_changes[:]

        if extract_melody:
            seq = strip_to_melody(seq)

        melody_tensor = config.data_converter._melody_converter.to_tensors(seq)
        melody_tensor = melody_tensor.outputs[0]

        # Pad sequence to length 256, batch size of 1
        pad = max(0, 256 - melody_tensor.shape[0])
        melody_tensor = np.pad(melody_tensor, [(0, pad), (0, 0)], 'constant')
        melodies.append(melody_tensor)

    latent_codes = surrogate_encoder.predict(melodies)
    decoded_sequences = musicvae.decode(latent_codes, length=64,
                                        temperature=temperature)

    out_sequences = []
    if stitch:
        for melody, accompaniment in zip(sequences, decoded_sequences):
            out = remove_melody(accompaniment)
            out.notes.extend(melody.notes)
            out_sequences.append(out)
    else:
        out_sequences = decoded_sequences

    return out_sequences
