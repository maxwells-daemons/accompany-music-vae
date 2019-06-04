'''
Code to generate accompaniments.

Uses a trained surrogate encoder and a pretrained decoder to perform inference.

Functions
---------
generate_accompaniment() : NoteSequence
    Generate accompaniment for a note sequence.

Examples
--------
```
model = keras.models.load_model(...)
song = './data/lmd_clean/raw/Michael Jackson/Beat It.mid'
accompaniments = generate_accompaniments(song, model)
```
'''


import os
from copy import deepcopy

import numpy as np
import magenta.music as mm
from magenta.models.music_vae import configs, trained_model
from magenta.music.sequences_lib import concatenate_sequences

from constants import (MUSICVAE_MODEL_NAME, MUSICVAE_MODEL_PATH, TIMESTEPS)
from utils import (strip_to_melody, remove_melody)


def generate_accompaniment(seq, surrogate_encoder, musicvae=None,
                           stitch=True, extract_melody=True,
                           remove_controls=True, temperature=0.1):
    '''
    Generate accompaniment for an input sequence.

    Parameters
    ----------
    seq : str path to midi or NoteSequence
        The input sequence.
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
    NoteSequence
        The input sequence along with generated accompaniment.
    '''

    config = configs.CONFIG_MAP[MUSICVAE_MODEL_NAME]
    melody_converter = config.data_converter._melody_converter

    musicvae = musicvae or trained_model.TrainedModel(
        config, batch_size=16,
        checkpoint_dir_or_path=os.path.join(MUSICVAE_MODEL_PATH,
                                            MUSICVAE_MODEL_NAME + '.ckpt')
    )

    # If the sequence is provided as a MIDI path, load it
    if isinstance(seq, str):
        midi = None
        with open(seq, 'rb') as midi_file:
            midi = midi_file.read()
        seq = mm.midi_to_sequence_proto(midi)

    if remove_controls:
        del seq.tempos[1:]
        del seq.time_signatures[1:]
        del seq.control_changes[1:]

    if extract_melody:
        seq = strip_to_melody(seq)

    # Convert the input NoteSequence to a single-instrument tensor
    melody_tracks = melody_converter.to_tensors(seq).outputs
    instrument_counts = [np.sum(melody_tracks[i][:, 1:])
                         for i in range(len(melody_tracks))]
    instrument_idx = np.argmax(instrument_counts)
    melody_tensor = melody_tracks[instrument_idx]

    # Slice the melody into non-overlapping windows
    windows = [melody_tensor[i * TIMESTEPS:(i+1) * TIMESTEPS, :]
               for i in range(melody_tensor.shape[0] // TIMESTEPS + 1)]
    windows[-1] = np.pad(windows[-1],
                         [(0, max(0, TIMESTEPS - windows[-1].shape[0])),
                          (0, 0)],
                         mode='constant')
    windows_stacked = np.stack(windows)

    # Perform inference
    latent_codes = surrogate_encoder.predict(windows_stacked)
    decoded_sequences = musicvae.decode(latent_codes, temperature=temperature)
    decoded = concatenate_sequences(decoded_sequences)

    # Stitch the original melody and the new accompaniment together.
    if stitch:
        melody_tensor_padded = np.stack(windows_stacked, axis=0)
        melody_padded = concatenate_sequences(
            melody_converter.to_notesequences(melody_tensor_padded))
        out = remove_melody(decoded)
        out.MergeFrom(melody_padded)
        return out

    return decoded
