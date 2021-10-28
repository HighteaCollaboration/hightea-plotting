import pathlib
import numpy as np
import itertools
import warnings
from ._MeasurementTools import MeasurementTools
from .run import Run

def convert_to_Run(mt: MeasurementTools, file=0, **kwargs):
    run = Run()
    info = {}

    # Get file name
    fileid = file
    file = mt.files[fileid][0]
    info['file'] = file

    # Get observable
    obs = kwargs.get('obs',0)
    obslist = mt.extractObservables(fileid)
    if isinstance(obs, str):
        matchlist = [o for o in obslist if obs in o]
        if (matchlist):
            obs = matchlist[0]
            if len(matchlist) > 1:
                warnings.warn(f'several observables match, using:\n"{obs}"')
        else:
            raise Exception(f'No observables match "{obs}"')
    else:
        obs = obslist[obs]
    histid = kwargs.get('hist',0)

    # Get variations
    available_setups = mt.extractSetups(fileid)
    setupids = kwargs.get('setups',np.arange(len(available_setups)))
    setupid = setupids[0]

    # Other options
    verbose = kwargs.get('verbose',0)
    withOUF = kwargs.get('withOUF',False)

    if (verbose):
        print(f'converting {file} for "{obs}" (#{histid})')

    # Extract basic histogram information
    hist = mt.extractHistograms(fileid, obs)[histid]
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

    info['obs'] = obs
    info['histid'] = histid
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

