

import logging

def initLogger(logfile):
    # resetting log file (if appending is preferred, comment this out; in DEBUG mode, log file is large)
    with open(logfile,'w'):
        pass
    logging.basicConfig(filename=logfile, level=logging.DEBUG,format='%(asctime)s %(message)s')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s %(asctime)s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
