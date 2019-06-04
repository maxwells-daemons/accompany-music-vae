'''
Project-wide utilities.

Functions
---------
is_melody(Note) : bool
strip_to_melody(NoteSequence) : NoteSequence
remove_melody(NoteSequence) : NoteSequence
'''


from copy import deepcopy


def is_melody(note):
    return note.program <= 31 and not note.is_drum


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
    melody_notes = list(filter(lambda note: is_melody(note),
                               # note.instrument == 1 or note.instrument == 0,
                               sequence.notes[:]))
    del melody.notes[:]
    melody.notes.extend(melody_notes)
    return melody


def remove_melody(sequence):
    '''
    Strip a NoteSequence of the melody.

    Parameters
    ----------
    sequence : NoteSequence
        The input sequence.

    Returns
    -------
    NoteSequence
        The input sequence's non-melody as a NoteSequence.
    '''
    not_melody = deepcopy(sequence)
    not_melody_notes = list(filter(lambda note: not is_melody(note),
                                   sequence.notes[:]))
    del not_melody.notes[:]
    not_melody.notes.extend(not_melody_notes)
    return not_melody
