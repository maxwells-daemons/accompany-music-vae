'''
Project-wide constants.

Attributes
----------
DATA_PATH : str path
   Base path of data files.
CHECKPOINT_PATH : str path
    Base path of model checkpoint files.
MUSICVAE_MODEL_PATH : str path
    Base path of pretrained MusicVAE models.
MUSICVAE_MODEL_PATH : str
    Name of MusicVAE trio model used.
LOG_PATH : str path
    Base path of log files.

TIMESTEPS : int
    How many timesteps are used for a training sequence.
DIM_LATENT : int
    Dimensionality of the latent space.
DIM_MELODY : int
    How many categories are used for modeling melody notes.
DIM_BASS : int
    How many categories are used for modeling bass notes.
DIM_DRUMS : int
    How many categories are used for modeling drum notes.
'''


DATA_PATH = './data/'
CHECKPOINT_PATH = './models/checkpoints/'
MUSICVAE_MODEL_PATH = './models/pretrained/'
MUSICVAE_MODEL_NAME = 'hierdec-trio_16bar'
LOG_PATH = './logs/'

TIMESTEPS = 256
DIM_LATENT = 512
DIM_MELODY = 90
DIM_BASS = 90
DIM_DRUMS = 512
DIM_TRIO = DIM_MELODY + DIM_BASS + DIM_DRUMS
