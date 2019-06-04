'''
Turn a tfrecord of NoteSequences into an HDF5 dataset with instruments split
and the pretrained trio mode's latent vectors.
'''

from copy import deepcopy
from time import time
import itertools as it

import click
import logging
from pprint import pformat

import numpy as np
import h5py

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)  # noqa

import magenta.music as mm
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel

from constants import (TIMESTEPS, DIM_MELODY, DIM_BASS, DIM_DRUMS, DIM_TRIO)

# Constants
MODEL_NAME = 'hierdec-trio_16bar'


@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path(exists=False))
@click.option('--include_all_instruments', type=bool, default=False)
@click.option('--chunk_size', type=click.IntRange(min=1), default=128,
              help='Number of MIDI files to read at once.')
@click.option('--buffer_size', type=click.IntRange(min=1), default=50000,
              help='Number of examples to make room for at a time.')
@click.option('--batch_size', type=click.IntRange(min=1), default=256,
              help='Batch size for the pretrained model.')
@click.option('--checkpoint', type=click.Path(),
              default='./models/pretrained/{}.ckpt'.format(MODEL_NAME),
              help='Checkpoint to use for the pretrained model.')
@click.option('--log_period', type=click.IntRange(min=0), default=1,
              help='How many chunks pass between logging lines.')
@click.option('--log_file', type=click.Path(),
              default='logs/split_dataset.log')
def main(input_file, output_file,
         include_all_instruments, chunk_size, buffer_size, batch_size,
         checkpoint, log_period, log_file):
    args = locals()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log = logging.getLogger(__name__)
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)

    log.info('Generating melody dataset with args:\n' + pformat(args))
    total_start_time = time()
    ns_gen = mm.note_sequence_io.note_sequence_record_iterator(input_file)
    ns_iter = iter(ns_gen)
    config = configs.CONFIG_MAP[MODEL_NAME]
    trio_converter = config.data_converter

    log.debug('Creating HDF5 store...')
    start_time = time()
    with h5py.File(output_file, 'w') as data_file:
        dataset_size = buffer_size
        ds_melody = data_file.create_dataset(
            'melody',
            (dataset_size, TIMESTEPS, DIM_MELODY),
            maxshape=(None, TIMESTEPS, DIM_MELODY),
            dtype=np.bool
        )
        ds_code = data_file.create_dataset(
            'code',
            (dataset_size, config.hparams.z_size),
            maxshape=(None, config.hparams.z_size),
            dtype=np.float32
        )

        if include_all_instruments:
            ds_trio = data_file.create_dataset(
                'trio',
                (dataset_size, TIMESTEPS, DIM_TRIO),
                maxshape=(None, TIMESTEPS, DIM_TRIO),
                dtype=np.bool
            )
            ds_bass = data_file.create_dataset(
                'bass',
                (dataset_size, TIMESTEPS, DIM_BASS),
                maxshape=(None, TIMESTEPS, DIM_BASS),
                dtype=np.bool
            )
            ds_drums = data_file.create_dataset(
                'drums',
                (dataset_size, TIMESTEPS, DIM_DRUMS),
                maxshape=(None, TIMESTEPS, DIM_DRUMS),
                dtype=np.bool
            )

        log.debug('Done creating HDF5 store (time: {0:.1f}s)'
                  .format(time() - start_time))

        log.debug('Loading model...')
        start_time = time()
        model = TrainedModel(config, batch_size=batch_size,
                             checkpoint_dir_or_path=checkpoint)
        log.debug('Done loading model (time: {0:.1f}s)'
                  .format(time() - start_time))

        log.info('Beginning dataset creation...')
        i_chunk = 0
        i_example = 0
        try:
            while True:
                i_chunk += 1
                log.disabled = i_chunk % log_period != 0 or not log_period
                chunk_time = time()

                log.debug('Processing a chunk of NoteSequences...')
                start_time = time()

                note_sequences = list(it.islice(ns_iter, chunk_size))
                if not note_sequences:
                    break

                trio_tensors = map(
                    lambda seq: trio_converter.to_tensors(seq).outputs,
                    note_sequences
                )
                trio_tensors = it.chain.from_iterable(trio_tensors)
                trio_tensors = list(
                    filter(lambda t: t.shape == (TIMESTEPS, DIM_TRIO),
                           trio_tensors)
                )

                # Ensure an example doesn't overflow the allocated space
                trio_tensors = trio_tensors[:buffer_size]
                n_tensors = len(trio_tensors)
                i_last = n_tensors + i_example

                melody_tensors = list(map(lambda t: t[:, :DIM_MELODY],
                                          trio_tensors))

                if include_all_instruments:
                    bass_tensors = list(map(
                        lambda t: t[:, DIM_MELODY:DIM_MELODY + DIM_BASS],
                        trio_tensors
                    ))
                    drums_tensors = list(map(lambda t: t[:, -DIM_DRUMS:],
                                             trio_tensors))

                log.debug('Done processing NoteSequences (time: {0:.1f}s)'
                          .format(time() - start_time))

                log.debug('Running encoder...')
                start_time = time()
                _, codes, _ = model.encode_tensors(deepcopy(trio_tensors),
                                                   [TIMESTEPS] * n_tensors)
                log.debug('Done running encoder (time: {0:.1f}s)'
                          .format(time() - start_time))

                if i_last >= dataset_size:
                    dataset_size += buffer_size
                    log.info('Resizing datasets to size:', dataset_size)
                    ds_melody.resize((dataset_size, TIMESTEPS, DIM_MELODY))
                    ds_code.resize((dataset_size, config.hparams.z_size))

                    if include_all_instruments:
                        ds_trio.resize((dataset_size, TIMESTEPS, DIM_TRIO))
                        ds_bass.resize((dataset_size, TIMESTEPS, DIM_BASS))
                        ds_drums.resize((dataset_size, TIMESTEPS, DIM_DRUMS))

                log.debug('Writing examples to HDF5...')
                start_time = time()
                ds_melody[i_example:i_last, :, :] = np.array(melody_tensors)
                ds_code[i_example:i_last, :] = np.array(codes)

                if include_all_instruments:
                    ds_trio[i_example:i_last, :, :] = np.array(trio_tensors)
                    ds_bass[i_example:i_last, :, :] = np.array(bass_tensors)
                    ds_drums[i_example:i_last, :, :] = np.array(drums_tensors)

                log.debug('Done writing examples to HDF5 (time: {0:.1f}s)'
                          .format(time() - start_time))

                i_example += n_tensors

                log.info(('Chunk {0} wrote {1} examples ' +
                         '(total: {2}; time: {3:.1f}s)')
                         .format(i_chunk, n_tensors, i_example,
                                 time() - chunk_time))
        except StopIteration:
            pass

    log.debug('Finished writing data')
    log.debug('Resizing datasets...')
    dataset_size = i_example
    ds_melody.resize((dataset_size, TIMESTEPS, DIM_MELODY))
    ds_code.resize((dataset_size, config.hparams.z_size))
    if include_all_instruments:
        ds_trio.resize((dataset_size, TIMESTEPS, DIM_TRIO))
        ds_bass.resize((dataset_size, TIMESTEPS, DIM_BASS))
        ds_drums.resize((dataset_size, TIMESTEPS, DIM_DRUMS))
    log.debug('Done resizing datasets...')

    total_time = time() - total_start_time
    log.info('Finished creating HDF5 dataset')
    log.info('Total examples: {}'.format(i_example))
    log.info('Total chunks: {}'.format(i_chunk))
    log.info('Total time: {0:.1f}s'.format(total_time))
    log.info('Done!')


if __name__ == '__main__':
    main()
