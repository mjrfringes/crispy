import logging
import platform
import sys
import traceback

log_dict = {}


class crispyLogger(logging.getLoggerClass()):
    """
    This is the advanced logging object used for crispy.
    It is almost a carbon copy of the CHARIS DRP logger routine.
    It inherits from the standard
    Python library 'logging' and provides added features.
    The default log level for the output file will be 1, ie ALL messages;
    while the default for the screen will be INFO, and can be changed easily
    using the setStreamLevel(lvl) member function.
    """

    def setStreamLevel(self, lvl=20):
        """Set/change the level for the stream handler for a logging object.
        Any file handlers will be left alone.
        All messages of a higher severity level than 'lvl' will be printed
        to the screen.

        Args:
            lvl (int): The severity level of messages printed to the screen with
                    the stream handler, default = 20.

        +---------------------+----------------------+
        |    Standard Levels  |        New Levels    |
        +---------------+-----+----------------+-----+
        |    Name       |Level|  Name          |Level|
        +===============+=====+================+=====+
        |CRITICAL       |  50 | MAINCRITICAL   | 80  |
        +---------------+-----+----------------+-----+
        |ERROR          |  40 | MAINERROR      | 75  |
        +---------------+-----+----------------+-----+
        |WARNING        |  30 | MAINWARNING    | 70  |
        +---------------+-----+----------------+-----+
        |INFO           |  20 | MAININFO       | 65  |
        +---------------+-----+----------------+-----+
        |DEBUG          |  10 | MAINDEBUG      | 60  |
        +---------------+-----+----------------+-----+
        |NOTSET         |  0  | PRIMCRITICAL   | 55  |
        +---------------+-----+----------------+-----+
        |               |     | PRIMERROR      | 49  |
        +               +     +----------------+-----+
        |               |     | PRIMWARNING    | 45  |
        +               +     +----------------+-----+
        |               |     | PRIMINFO       | 39  |
        +               +     +----------------+-----+
        |               |     | PRIMDEBUG      | 35  |
        +               +     +----------------+-----+
        |               |     | TOOLCRITICAL   | 29  |
        +               +     +----------------+-----+
        |               |     | TOOLERROR      | 25  |
        +               +     +----------------+-----+
        |               |     | TOOLWARNING    | 19  |
        +               +     +----------------+-----+
        |               |     | TOOLINFO       | 15  |
        +               +     +----------------+-----+
        |               |     | TOOLDEBUG      | 9   |
        +               +     +----------------+-----+
        |               |     | SUMMARY        | 5   |
        +---------------+-----+----------------+-----+

        """
        verbose = False
        if verbose:
            print('Changing logging level to ' + repr(lvl))
        # Kill off the old handlers and reset them with the setHandlers func
        for i in range(0, len(self.handlers)):
            h = self.handlers[i]
            if isinstance(h, logging.FileHandler):
                if verbose:
                    print('filehandler type')
            elif isinstance(h, logging.StreamHandler):
                #print 'stream handler type'
                if verbose:
                    print('removing handler %s' % str(h))
                self.removeHandler(h)
                break
            if verbose:
                print('%d more to go' % len(self.handlers))
        addStreamHandler(self, lvl)

    # Add the new log levels needed for the 3 tier hierarchy plus the summary
    # level to the logging object.
    # Levels for the 'main', or top, tier.
    MAINCRITICAL = 80
    logging.addLevelName(MAINCRITICAL, 'MAINCRITICAL')

    def maincritical(self, msg, lvl=MAINCRITICAL, *args, **kws):
        self.log(lvl, msg, *args, **kws)
    logging.Logger.maincritical = maincritical
    MAINERROR = 75
    logging.addLevelName(MAINERROR, 'MAINERROR')

    def mainerror(self, msg, lvl=MAINERROR, *args, **kws):
        self.log(lvl, msg, *args, **kws)
    logging.Logger.mainerror = mainerror
    MAINWARNING = 70
    logging.addLevelName(MAINWARNING, 'MAINWARNING')

    def mainwarning(self, msg, lvl=MAINWARNING, *args, **kws):
        self.log(lvl, msg, *args, **kws)
    logging.Logger.mainwarning = mainwarning
    MAININFO = 65
    logging.addLevelName(MAININFO, 'MAININFO')

    def maininfo(self, msg, lvl=MAININFO, *args, **kws):
        self.log(lvl, msg, *args, **kws)
    logging.Logger.maininfo = maininfo
    MAINDEBUG = 60
    logging.addLevelName(MAINDEBUG, 'MAINDEBUG')

    def maindebug(self, msg, lvl=MAINDEBUG, *args, **kws):
        self.log(lvl, msg, *args, **kws)
    logging.Logger.maindebug = maindebug
    # Levels for the 'prims' tier.
    PRIMCRITICAL = 55
    logging.addLevelName(PRIMCRITICAL, 'PRIMCRITICAL')

    def primcritical(self, msg, lvl=PRIMCRITICAL, *args, **kws):
        self.log(lvl, msg, *args, **kws)
    logging.Logger.primcritical = primcritical
    PRIMERROR = 49
    logging.addLevelName(PRIMERROR, 'PRIMERROR')

    def primerror(self, msg, lvl=PRIMERROR, *args, **kws):
        self.log(lvl, msg, *args, **kws)
    logging.Logger.primerror = primerror
    PRIMWARNING = 45
    logging.addLevelName(PRIMWARNING, 'PRIMWARNING')

    def primwarning(self, msg, lvl=PRIMWARNING, *args, **kws):
        self.log(lvl, msg, *args, **kws)
    logging.Logger.primwarning = primwarning
    PRIMINFO = 39
    logging.addLevelName(PRIMINFO, 'PRIMINFO')

    def priminfo(self, msg, lvl=PRIMINFO, *args, **kws):
        self.log(lvl, msg, *args, **kws)
    logging.Logger.priminfo = priminfo
    PRIMDEBUG = 35
    logging.addLevelName(PRIMDEBUG, 'PRIMDEBUG')

    def primdebug(self, msg, lvl=PRIMDEBUG, *args, **kws):
        self.log(lvl, msg, *args, **kws)
    logging.Logger.primdebug = primdebug
    # Levels for the 'tools' tier.
    TOOLCRITICAL = 29
    logging.addLevelName(TOOLCRITICAL, 'TOOLCRITICAL')

    def toolcritical(self, msg, lvl=TOOLCRITICAL, *args, **kws):
        self.log(lvl, msg, *args, **kws)
    logging.Logger.toolcritical = toolcritical
    TOOLERROR = 25
    logging.addLevelName(TOOLERROR, 'TOOLERROR')

    def toolerror(self, msg, lvl=TOOLERROR, *args, **kws):
        self.log(lvl, msg, *args, **kws)
    logging.Logger.toolerror = toolerror
    TOOLWARNING = 19
    logging.addLevelName(TOOLWARNING, 'TOOLWARNING')

    def toolwarning(self, msg, lvl=TOOLWARNING, *args, **kws):
        self.log(lvl, msg, *args, **kws)
    logging.Logger.toolwarning = toolwarning
    TOOLINFO = 15
    logging.addLevelName(TOOLINFO, 'TOOLINFO')

    def toolinfo(self, msg, lvl=TOOLINFO, *args, **kws):
        self.log(lvl, msg, *args, **kws)
    logging.Logger.toolinfo = toolinfo
    TOOLDEBUG = 9
    logging.addLevelName(TOOLDEBUG, 'TOOLDEBUG')

    def tooldebug(self, msg, lvl=TOOLDEBUG, *args, **kws):
        self.log(lvl, msg, *args, **kws)
    logging.Logger.tooldebug = tooldebug
    # Level for the 'summary' info used for the main log file and the
    # special summary file and the fits headers.
    SUMMARY = 5
    logging.addLevelName(SUMMARY, 'SUMMARY')

    def summary(self, msg, lvl=SUMMARY, *args, **kws):
        self.log(lvl, msg, *args, **kws)
    logging.Logger.summary = summary


def getLogger(name='generalLoggerName', lvl=20, addFH=True, addSH=True):
    """This will either return the logging object already
    instantiated, or instantiate a new one and return it.
    **Use this function to both create and return any logger** to avoid
    accidentally adding additional handlers by using the setUpLogger function
    instead.

    Args:
        name (str): The name for the logging object and
                    name.log will be the output file written to disk.
        lvl (int): The severity level of messages printed to the screen with
                    the stream handler, default = 20.
        addFH (boolean): Add a file handler to this logger?  Default severity
                         level for it will be 1, and it will be named following
                         name+'.log'.  Default = True.
        addSH (boolean): Add a stream handler to this logger? Severity set with
                        the lvl argument.  Default = True.
    Returns:
        log (CharisLogger object): A CharisLogger object that was either
                                  freshly instantiated or determined to
                                  already exist, then returned.
    """
    log = False
    verbose = False
    try:
        log = log_dict[name]
        if verbose:
            print (repr(log_dict))
            print ('found a log by the name already exists so returning it')
    except BaseException:
        if verbose:
            print ('No logger object found so creating one with the name ' + name)
        log = setUpLogger(name, lvl, addFH, addSH)
    return log


def setUpLogger(name='generalLoggerName', lvl=20, addFH=True, addSH=True):
    """ This function is utilized by getLogger to set up a new logging object.
    It will have the default name 'generalLoggerName' and stream handler level
    of 20 unless redefined in the function call.
    NOTE:
    If a file handler is added, it will have the lowest severity level by
    default (Currently no need for changing this setting, so it will stay
    this way for now).  Remember that any messages will be passed up to any
    parent loggers, so children do not always need their own file handler.

    Args:
        name (str): The name for the logging object and
                    name.log will be the output file written to disk.
        lvl (int): The severity level of messages printed to the screen with
                    the stream handler, default = 20.
        addFH (boolean): Add a file handler to this logger?  Default severity
                         level for it will be 1, and it will be named following
                         name+'.log'.  Default = True.
        addSH (boolean): Add a stream handler to this logger? Severity set with
                        the lvl argument.  Default = True.
    Returns:
        log (crispyLogger object): A crispyLogger object that was freshly
                                   instantiated.
    """
    logging.setLoggerClass(crispyLogger)
    log = logging.getLogger(name)
    log_dict[name] = log
    log.setLevel(1)
    # add the requested handlers to the log
    if addFH:
        addFileHandler(log, lvl=1)
    # make a stream handler
    if addSH:
        addStreamHandler(log, lvl)
    return log


def addFileHandler(log, lvl=1):
    """
    This function will add a file handler to a log with the provided level.

    Args:
        log (crispyLogger object): A crispyLogger object that was freshly
                                   instantiated.
        lvl (int): The severity level of messages printed to the file with
                    the file handler, default = 1.
    """
    verbose = False
    if verbose:
        print('Setting FileHandler level to ' + str(lvl))
    fh = logging.FileHandler(log.name + '.log')
    fh.setLevel(lvl)
    frmtString = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    fFrmt = logging.Formatter(frmtString)
    fh.setFormatter(fFrmt)
    log.addHandler(fh)


def addStreamHandler(log, lvl=20):
    """
    This function will add a stream handler to a log with the provided level.

    Args:
        log (crispyLogger object): A crispyLogger object that was freshly
                                   instantiated.
        lvl (int): The severity level of messages printed to the screen with
                    the stream handler, default = 20.
    """
    verbose = False
    if verbose:
        print('Setting StreamHandler level to ' + str(lvl))
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(lvl)
    sFrmt = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    sh.setFormatter(sFrmt)
    log.addHandler(sh)


def addFitsStyleHandler(log):
    """
    This function will add a file handler with a string format ideal for
    directly loading into a FITS header.

    Args:
        log (crispyLogger object): A crispyLogger object that was freshly
                                   instantiated.
    """
    fitsFhLevel = 1
    verbose = False
    if verbose:
        print('Setting FITS FileHandler level to ' + str(fitsFhLevel))
    fh2 = logging.FileHandler(log.name + '.fitsFormat.log')
    fh2.setLevel(fitsFhLevel)
    frmtString2 = '%(asctime)s - %(message)s'
    fFrmt2 = logging.Formatter(frmtString2)
    fh2.setFormatter(fFrmt2)
    # add the Handler to the logger
    log.addHandler(fh2)
