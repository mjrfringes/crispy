import logging

def initLogger(logfile,levelConsole=logging.INFO,levelLogFile=logging.DEBUG):
    with open(logfile,'w'):
        pass
    logging.basicConfig(filename=logfile, level=levelLogFile,format='%(asctime)s %(message)s')
    console = logging.StreamHandler()
    console.setLevel(levelConsole)
    formatter = logging.Formatter('%(asctime)s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
