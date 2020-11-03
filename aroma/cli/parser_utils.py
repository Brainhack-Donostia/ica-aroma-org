"""Utility functions for CLI parsers."""
import os.path as op


def is_valid_file(parser, arg):
    """
    Check if argument is existing file.
    """
    if not op.isfile(arg) and arg is not None:
        parser.error('The file {0} does not exist!'.format(arg))

    return arg


def is_valid_path(parser, arg):
    """
    Check if argument is existing directory.
    """
    if not op.isdir(arg) and arg is not None:
        parser.error('The folder {0} does not exist!'.format(arg))

    return arg
