import numpy as np
from keras.utils import Sequence


class HDF5Sequence(Sequence):
    '''
    A Keras Sequence class which generates batches of pairs of columns
    read from an HDF5 file.
    '''
    def __init__(self, data_file, batch_size, columns=['melody', 'code'],
                 index_path=None):
        '''
        Initialize a HDF5Sequence.

        Parameters
        ----------
        data_file : h5py File object
            The HDF5 file containing the data. Must be readable and should not
            be closed while the Sequence is in use.
        batch_size : int
            The batch size.
        columns : list of str
            The columns of the HDF5 file to be returned as data.
            Must not be empty.
        index_path : str path to csv, optional
            Path to a csv containing an ordering of indices into the rows
            of the HDF5 data file. If None, then all rows are used in order.
        '''
        self.batch_size = batch_size
        self._datasets = [data_file[col] for col in columns]
        self._shapes = [(batch_size,) + ds.shape[1:] for ds in self._datasets]
        self._index = np.loadtxt(index_path, dtype=int) if index_path \
            else np.arange(len(self._datasets[0]))

    def __len__(self):
        return int(np.ceil(len(self._index) / float(self.batch_size)))

    def __getitem__(self, index):
        '''
        Retrieve one batch of data.

        Parameters
        ----------
        index : int
            Which batch to retrieve.

        Returns
        -------
        tuple
            The columns of the data file at index.
        '''
        indices = self._index[index:index + self.batch_size]
        outs = []
        for ds, shape in zip(self._datasets, self._shapes):
            outs.append(np.zeros(shape))
            for j, i in enumerate(indices):
                outs[-1][j] = ds[i]
        return tuple(outs)
