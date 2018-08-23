# Exposure time calculator for coronagraphs
# Based on Bijan Nemati's yield speadsheet


import pandas as pd
import numpy as np
from astropy.table import Table,QTable
from astropy.io import ascii
import astropy.units as u
from crispy.tools.initLogger import getLogger
log = getLogger('crispy')
import os

from scipy.interpolate import interp1d


path = os.path.dirname(__file__)

def loadCGs(filename=None):
    '''
    Load coronagraph tables
    '''
    def p2f(x):
        return float(x.strip('%'))/100
    if filename is None:
        fname = path+'/Inputs/ETC/CGs.csv'
    else:
        fname = filename
    c = pd.read_csv(fname,converters={'BW':p2f},skip_blank_lines=True)
    cgs = QTable.from_pandas(c)
    cgs.add_index('CG')
    return cgs
    
def loadCoronagraph(name):
    '''
    Load single coronagraph description
    '''
    if not isinstance(name,basestring):
        log.error('Need a string for the coronagraph name')
        raise
    else:
        coron = pd.read_csv(path+'/Inputs/ETC/'+name+'.csv',skip_blank_lines=True)
        coron = coron.dropna()
        corono = QTable.from_pandas(coron)
        corono.add_index('r(arcsec)')
        corono['r(arcsec)'].unit = u.arcsec
        corono['area(sq_arcsec'].unit = u.arcsec**2
        return corono

def loadScenarios(filename=None):
    '''
    Load various observing scenarios
    '''
    if filename is None:
        fname = path+'/Inputs/ETC/Scenarios.csv'
    else:
        fname = filename
    scens = pd.read_csv(fname,skip_blank_lines=True)
    scenarios = QTable.from_pandas(scens)
    
    # two indexes
    scenarios.add_index('Scenario')
    scenarios.add_index('Coronagraph')
    scenarios['Center lam'].unit = u.nm
    scenarios['t integ, hrs'].unit = u.h
    scenarios['Years at L2'].unit = u.year
    scenarios['Ref  Dmag'].unit = u.mag
    return scenarios
    
def loadPlanets(filename=None):
    '''
    Load planet database
    '''
    if filename is None:
        fname = path+'/Inputs/ETC/Planets.csv'
    else:
        fname = filename
    pl = pd.read_csv(fname,skip_blank_lines=True)
    planets = QTable.from_pandas(pl)
    planets['V'].unit=u.mag
    planets['A'].unit=u.au
    planets['DIST'].unit=u.pc
    planets['Rp / R_J'].unit=u.R_jup
    planets.add_index(['NAME'])
    return planets

def loadSpectra(filename=None):
    '''
    Load stellar spectra database
    '''

    if filename is None:
        fname = path+'/Inputs/ETC/Spectra.csv'
    else:
        fname = filename
    st = pd.read_csv(fname,skip_blank_lines=True,)
    stars = QTable.from_pandas(st)
    stars['Wavelength (m) '].unit = u.m
    stars.rename_column('Wavelength (m) ', 'Wavelength')
    stars['E_ph (J)'].unit = u.J
    stars.rename_column('E_ph (J)', 'E_ph')
    cols = ['a0v','a5v','f5v','g0v','g5v','k0v','k5v','m0v','m5v']
    for col in cols:
        stars[col].unit = u.W/u.m**2/u.m
    return stars

class Telescope(object):
    def __init__(self,  D = 2.37*u.m,
                    obs = 0.32,
                    strut_obs = 0.07):
        self.diam = D
        self.cobscuration = obs
        self.sobscuration = strut_obs
    
    @property
    def clearApFrac(self):
        return (1.-self.sobscuration)*(1.-self.cobscuration**2)

    @property
    def Aeff(self):
        return 0.25*np.pi*self.diam**2*self.clearApFrac
                
    @property
    def table(self):
        _diam = {'Name': 'Diameter',
                    'Value': self.diam,
                    'Comment': 'Primary mirror diameter',
                    'Formula': '',
                    'Symbol': r'$$D_\mathrm{PM}$$'}
        _cobscuration = {'Name': 'Central obscuration',
                            'Value': self.cobscuration,
                            'Comment': 'pct of primary mirror diameter',
                            'Formula': '',
                            'Symbol': r'$$\mathrm{oc}$$'}
        _sobscuration = {'Name': 'Strut obscuration',
                            'Value': self.sobscuration,
                            'Comment': 'pct of primary mirror area',
                            'Formula': '',
                            'Symbol': r'$$\mathrm{os}$$'}
        _clearApFrac = {'Name': 'Clear aperture fraction',
                    'Value': self.clearApFrac,
                    'Comment': 'After obscuration by secondary mirror and struts',
                    'Formula': r'$$(1-%s)\times (1-%s^2)$$' % (_sobscuration['Symbol'].replace('$',''),
                                                           _cobscuration['Symbol'].replace('$','')),
                    'Symbol': r'$$\mathrm{obsc}$$'}
        _Aeff = {'Name': 'Effective aperture',
                    'Value': self.Aeff,
                    'Comment': 'After obscuration by secondary mirror and struts',
                    'Formula': r'$$%s \times%s$$' % (_diam['Symbol'].replace('$',''),
                                                _clearApFrac['Symbol'].replace('$','')),
                    'Symbol': r'$$A_\mathrm{eff}$$'}
        tab = QTable.from_pandas(pd.DataFrame([_diam,
                                               _cobscuration,
                                               _sobscuration,
                                               _clearApFrac,
                                               _Aeff]))
        tab.add_index('Name')
        return tab['Name','Value','Symbol','Formula','Comment']
    
#     def __repr__(self):
#         return ascii.write(self.table)
                
class Database(object):
    '''
    Gobal environment that loads all the required databases
    '''
    
    def __init__(self,
                CGsName = None,
                ScenariosName=None,
                PlanetsName=None,
                SpectraName=None,
                localZodi = 23*u.mag/u.arcsec**2,
                exoZodi = 22*u.mag/u.arcsec**2,
                nExoZodi = 1):
        print(__file__)
        self.CGs = loadCGs(CGsName)
        self.Scenarios = loadScenarios(ScenariosName)
        self.Spectra = loadSpectra(SpectraName)
        self.Planets = loadPlanets(PlanetsName)
        self.Telescope = Telescope()

#     @property
#     def computeCoronagraphParams(self,name):
#         '''
#         Compute the coronagraph parameters given the coronagraph name
#         '''
#         # load corresponding coronagraph
#         try:
#             corono = loadCoronagraph(name)
#             log.info('Successfully loaded coronagraph %s' % name)
#         except:
#             log.error('Error loading coronagraph')
#             raise
#         
#         # now form the parameter table
#         radius = {'Name': 'Radius',
#         'Value': corono.iloc[0]['r(arcsec)'],
#         'Comment': 'Radius at IWA',
#         'Formula': '',
#         'Symbol': '$$r_\mathrm{IWA}$$'}
#         wavel = {'Name': 'Design wavelength',
#                 'Value': (self.Telescope.diam['Value']* \ 
#                         corono.iloc[0]['r(arcsec)']* \
#                         corono.iloc[0]['r(lam/D)']).to(u.nm, equivalencies=u.dimensionless_angles()),
#                 'Comment': '',
#                 'Formula': r'',
#                 'Symbol': r'$$\lambda_\mathrm{d}$$'}
#         wavel = {'Name': 'Intensity',
#                 'Value': (self.Telescope.diam['Value']* \ 
#                         corono.iloc[0]['r(arcsec)']* \
#                         corono.iloc[0]['r(lam/D)']).to(u.nm, equivalencies=u.dimensionless_angles()),
#                 'Comment': '',
#                 'Formula': r'',
#                 'Symbol': r'$$\lambda_\mathrm{d}$$'}
# 
#                 
#         tab = QTable.from_pandas(pd.DataFrame([ radius,
#                                                 wavel]))
#         tab.add_index('Name')

#     coronagraphParams = property(getCoronagraphParams,doc="Coronagraph parameters")

db = Database()

def Yield(scenario):
    '''
    Calculate the planet yield for a given scenario
    '''
    # loop on all the planets and check how many are below the max exp time
    pass
    

def ETC(scenario,planet):
    '''
    Calculate the exposure time required to reach SNR on a given planet
    '''
    # calculate count rates of all components and thus exposure time
    pass

def Rates(scenario,planet):
    '''
    Calculate the count rates for all components
    '''
    pass

def Coronagraph(scenario,WA):
    '''
    Calculate coronagraph at working angle and the various throughput numbers
    '''
    pass
    
def PlanetParameters(scenario,planet):
    '''
    Derive the main planet parameters        
    '''
    _lam = db.Scenarios.loc[scenario]['Center lam']
    _R = db.Scenarios.loc[scenario]['R']
    _BW = db.Scenarios.loc[scenario]['BW']
    if _R>0:
        _dlam = _lam/_R
    else:
        _dlam = _BW*_lam
    _minlam = _lam-_dlam/2.0
    _maxlam = _lam+_dlam/2.0
    _SpType = db.Planets.loc[planet]['Use Spec']
    
    _wavArray = db.Spectra['Wavelength'].to(u.nm)
    _spectrum = db.Spectra[_SpType]
    _flux = interp1d(_wavArray,_spectrum)
    _integ = quad(_flux,_minlam,_maxlam)
    print(_lam)
    pass