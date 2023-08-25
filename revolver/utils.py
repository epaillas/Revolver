import numpy as np
import logging
import sys
import os
import time
import traceback


logger = logging.getLogger('Utils')


def setup_logging(level=logging.INFO, stream=sys.stdout, filename=None, filemode='w', **kwargs):
    """
    Set up logging.

    Parameters
    ----------
    level : string, int, default=logging.INFO
        Logging level.

    stream : _io.TextIOWrapper, default=sys.stdout
        Where to stream.

    filename : string, default=None
        If not ``None`` stream to file name.

    filemode : string, default='w'
        Mode to open file, only used if filename is not ``None``.

    kwargs : dict
        Other arguments for :func:`logging.basicConfig`.
    """
    # Cannot provide stream and filename kwargs at the same time to logging.basicConfig, so handle different cases
    # Thanks to https://stackoverflow.com/questions/30861524/logging-basicconfig-not-creating-log-file-when-i-run-in-pycharm
    if isinstance(level, str):
        level = {'info': logging.INFO, 'debug': logging.DEBUG, 'warning': logging.WARNING}[level.lower()]
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)

    t0 = time.time()

    class MyFormatter(logging.Formatter):

        def format(self, record):
            self._style._fmt = '[%09.2f] ' % (time.time() - t0) + ' %(asctime)s %(name)-28s %(levelname)-8s %(message)s'
            return super(MyFormatter, self).format(record)

    fmt = MyFormatter(datefmt='%m-%d %H:%M ')
    if filename is not None:
        mkdir(os.path.dirname(filename))
        handler = logging.FileHandler(filename, mode=filemode)
    else:
        handler = logging.StreamHandler(stream=stream)
    handler.setFormatter(fmt)
    logging.basicConfig(level=level, handlers=[handler], **kwargs)
    sys.excepthook = exception_handler

def exception_handler(exc_type, exc_value, exc_traceback):
    """Print exception with a logger."""
    # Do not print traceback if the exception has been handled and logged
    _logger_name = 'Exception'
    log = logging.getLogger(_logger_name)
    line = '=' * 100
    # log.critical(line[len(_logger_name) + 5:] + '\n' + ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)) + line)
    log.critical('\n' + line + '\n' + ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)) + line)
    if exc_type is KeyboardInterrupt:
        log.critical('Interrupted by the user.')
    else:
        log.critical('An error occured.')

def mkdir(dirname):
    """Try to create ``dirname`` and catch :class:`OSError`."""
    try:
        os.makedirs(dirname)  # MPI...
    except OSError:
        return


def sky_to_cartesian(data, cosmology):
    '''Converts ra, dec, redshift to cartesian coordinates.

    Parameters
    ----------
    data : array_like
        Array of ra, dec, redshift.
    cosmology : Cosmology
        Cosmology object.

    Returns
    -------
    cout : array_like
        Array of x, y, z coordinates.
    '''
    ra = data[:, 0]
    dec = data[:, 1]
    redshift = data[:, 2]

    if np.any(dec > 90):
        dec = 90 - dec

    dist = cosmology.ComovingDistance(redshift)
    x = dist * np.cos(dec * np.pi / 180) * np.cos(ra * np.pi / 180)
    y = dist * np.cos(dec * np.pi / 180) * np.sin(ra * np.pi / 180)
    z = dist * np.sin(dec * np.pi / 180)

    cout = np.c_[x, y, z]
    return cout


def cartesian_to_sky(data, cosmology):
    '''Converts cartesian coordinates to ra, dec, redshift.

    Parameters
    ----------
    data : array_like
        Array of x, y, z coordinates.
    cosmology : Cosmology
        Cosmology object.

    Returns
    -------
    cout : array_like
        Array of ra, dec, redshift.
    '''
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    dec = 90 - np.degrees(np.arccos(z / dist))
    ra = np.degrees(np.arctan2(y, x))
    ra[ra < 0] += 360
    redshift = cosmology.Redshift(dist)
    cout = np.c_[ra, dec, redshift]
    return cout