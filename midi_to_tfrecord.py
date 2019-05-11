'''
Convert all of the MIDI files under a directory into NoteSequences
and save as a tfrecord.
'''

import click


@click.command()
@click.argument('input_dir', type=click.Path())
@click.argument('output_file', type=click.Path())
def main(input_dir, output_file):
    # Import here to fail fast with wrong args
    from magenta.scripts import convert_dir_to_note_sequences as converters
    converters.convert_directory(input_dir, output_file, True)


if __name__ == '__main__':
    main()
