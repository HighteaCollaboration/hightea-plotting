import pathlib
import numpy as np
import itertools
import warnings
from ._MeasurementTools import MeasurementTools as MT
from .run import Run


def _list_histogram_setups(histograms):
    for i,hist in enumerate(histograms):
        edgesList = MT.histogramEdges(hist)
        smearing = MT.histogramSmearing(hist)
        dims = '*'.join([str(len(edges)-1) for edges in edgesList])
        binning = ' * '.join([f'({edges[0]}...{edges[-1]})'
                            for edges in edgesList])
        print(f'#{i}: [{dims}, smear={smearing:.1f}] : {binning}')


def convert_to_Run(mt: MT, file=0, **kwargs):
    _verbose = kwargs.get('verbose',0)
    _obs = kwargs.get('obs',0)
    _hist = kwargs.get('hist',0)

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

    # Extract basic histogram information
    histograms = mt.extractHistograms(fileid, _obs)

    if (_verbose):
        print(f'Loading {file} for:\n"{_obs}" (#{_hist})')
        _list_histogram_setups(histograms)

    hist = histograms[_hist]
    edgesList = mt.histogramEdges(hist)
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
    info['hist'] = _hist
    info['smearing'] = mt.histogramSmearing(hist)
    info['nevents'] = int(mt.files[fileid][1].find('nevents').text)
    run.update_info(**info)
    run.name = file

    run.make_differential()

    return run



def load_to_Run(xmlfile, **kwargs):
    mt = MT()
    mt.loadxml(xmlfile)
    return convert_to_Run(mt, 0, **kwargs)

