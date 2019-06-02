MODEL_NAME = 'bi_rnn_test'
MODEL_SAVE_PATH = './models/checkpoints/{}/'.format(MODEL_NAME)
CHECKPOINT_PATH = MODEL_SAVE_PATH + '{epoch:02d}-{val_loss:.4f}.hdf5'

TIMESTEPS = 256
DIM_MELODY = 90
DIM_LATENT = 512
DIM_TRIO = 692
