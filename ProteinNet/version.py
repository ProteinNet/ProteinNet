import os
import logging
from threading import Thread

__version__ = '0.0.1'

try: 
    os.envirion['OUTDATED_IGNORE'] = '1'
    from outdated import check_outdated
except ImportError:
    check_outdated = None

def check():
    try:
        is_outdated, latest = check_outdated('ProteinNet', __version__)
        if is_outdated:
            logging.warning(
                f'The ProteinNet package is out of date. Your version is '
                f'{__version__}, while the latest version is {latest}.')
    except Exception:
        pass

if check_outdated is not None:
    thread = Thread(target=check)
    thread.start()
