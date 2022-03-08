# accompany-music-vae
Using [Magenta's MusicVAE](https://github.com/tensorflow/magenta/tree/master/magenta/models/music_vae) to generate musical accompaniment.

Read [my writeup on my personal site](https://aidanswope.com/2020/accompaniment/), or [check out our poster](https://drive.google.com/file/d/1SYW0uhId39YXQMXmvfQQf49A6d1J4ooF/view?usp=sharing)!

# Setup

## Required libraries

```bash
pip install keras tensorflow-gpu magenta-gpu pyFluidSynth
```

## Data
1. Download the Lakh MIDI dataset here: [https://colinraffel.com/projects/lmd/](https://colinraffel.com/projects/lmd/). For our final model, we used `LMD-full`.
2. Convert the MIDI files to a `.tfrecord`, e.g. run 
`python data_utils/midi_to_tfrecord.py data/clean_midi/ data/clean_midi_tfrecord.tfrecord`
3. Convert the `.tfrecord` to a `.hdf5`, e.g. run
`python data_utils/tfrecord_to_hdf5.py data/clean_midi_tfrecord.tfrecord data/clean_hdf5.hdf5`

## Training
Use `train.py` to train a model.
For the default model, you can use it as a Click script, e.g. run `python train.py --batch_size 16 --epochs 10`.
For more advanced use (e.g. custom models) import it as a module.

## Running the model
See `Demo.ipynb` for examples of how to make predictions using `inference.py`.
