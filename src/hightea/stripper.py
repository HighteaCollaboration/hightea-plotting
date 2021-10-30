import pathlib
import numpy as np
import itertools
import warnings
from ._MeasurementTools import MeasurementTools
from .run import Run

def convert_to_Run(mt: MeasurementTools, file=0, **kwargs):
    _verbose = kwargs.get('verbose',0)
    _obs = kwargs.get('obs',0)
    _histid = kwargs.get('hist',0)

    run = Run()
    info = {}

    # Get file name
    fileid = file
    file = mt.files[fileid][0]
    info['file'] = file

    # Get observable
    obslist = mt.extractObservables(fileid)
    if (_verbose):
        print('Observables list: ')
        for i,o in enumerate(obslist):
            print(i,o)

    if isinstance(_obs, str):
        matchlist = [o for o in obslist if _obs in o]
        if (matchlist):
            _obs = matchlist[0]
            if len(matchlist) > 1:
                warnings.warn(f'several observables match, using:\n"{_obs}"')
        else:
            raise Exception(f'No observables match "{_obs}"')
    else:
        _obs = obslist[_obs]

    # Get variations
    available_setups = mt.extractSetups(fileid)
    setupids = kwargs.get('setups',np.arange(len(available_setups)))
    setupid = setupids[0]

    # Other options
    withOUF = kwargs.get('withOUF',False)

    if (_verbose):
        print(f'Loading {file} for:\n"{_obs}" (#{_histid})')

    # Extract basic histogram information
    hist = mt.extractHistograms(fileid, _obs)[_histid]
    edgesList = mt.histogramBins(hist)
    run.edges = edgesList
    v = mt.histogramValues(hist, withOUF=withOUF)
    nsetups = v.shape[-1]
    v = v.reshape((len(run.bins),nsetups))
    e = mt.histogramErrors(hist, withOUF=withOUF).reshape((len(run.bins),nsetups))
    p = mt.histogramHits(hist, withOUF=withOUF).reshape((len(run.bins)))

    run.values = v[:,setupids]
    run.errors = e[:,setupids]
    run.xsec = np.transpose(mt.extractXSections(fileid)[setupids,:,0])

    info['obs'] = _obs
    info['histid'] = _histid
    info['smearing'] = mt.histogramSmearing(hist)
    info['nevents'] = int(mt.files[fileid][1].find('nevents').text)
    run.update_info(**info)
    run.name = file

    run.make_differential()

    return run


def load_to_Run(xmlfile, **kwargs):
    mt = MeasurementTools()
    mt.loadxml(xmlfile)
    return convert_to_Run(mt, 0, **kwargs)

