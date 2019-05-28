# accompany-music-vae
Using [Magenta's MusicVAE](https://github.com/tensorflow/magenta/tree/master/magenta/models/music_vae) to generate musical accompaniment.

# Setup

## Required libraries

```bash
pip install keras tensorflow-gpu magenta-gpu pyFluidSynth
```

## Data
1. Download the Clean lakh midi dataset here: [https://colinraffel.com/projects/lmd/](https://colinraffel.com/projects/lmd/)
2. Convert the midi files to a `.tfrecord`, e.g. run 
`python data_utils/midi_to_tfrecord.py data/clean_midi/ data/clean_midi_tfrecord.tfrecord`
3. Convert the `.tfrecord` to a `.hdf5`, e.g. run
`python data_utils/tfrecord_to_hdf5.py data/clean_midi_tfrecord.tfrecord data/clean_hdf5.hdf5`