'''
Project-wide utilities.
'''

from copy import deepcopy


def strip_to_melody(sequence):
    '''
    Strip a NoteSequence of everything except the melody.

    Parameters
    ----------
    sequence : NoteSequence
        The input sequence.

    Returns
    -------
    NoteSequence
        The input sequence's melody as a NoteSequence.
    '''
    melody = deepcopy(sequence)
    melody_notes = list(filter(lambda note: not note.instrument,
                               sequence.notes[:]))
    del melody.notes[:]
    melody.notes.extend(melody_notes)
    return melody
