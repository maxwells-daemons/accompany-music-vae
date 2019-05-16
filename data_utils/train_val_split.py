'''
Generate csv files which encode a fixed training / validation split and
shuffling scheme.
'''

import os.path

import click
import h5py
import numpy as np


@click.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--output_path', type=click.Path(exists=True), default=None)
@click.option('--val_frac', type=float, default=0.2)
@click.option('--seed', type=int, default=1337)
def main(data_file, output_path, val_frac, seed):
    output_path = output_path or os.path.dirname(data_file)

    with h5py.File(data_file, 'r') as f:
        n_examples = f['melody'].shape[0]
    n_val = int(n_examples * val_frac)

    np.random.seed(seed)
    indices = np.random.permutation(n_examples)
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]

    np.savetxt(os.path.join(output_path, 'train_indices.csv'),
               train_indices, fmt='%i')
    np.savetxt(os.path.join(output_path, 'val_indices.csv'),
               val_indices, fmt='%i')


if __name__ == '__main__':
    main()
