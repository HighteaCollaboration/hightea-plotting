import pathlib
import numpy as np
import itertools
from ._MeasurementTools import MeasurementTools
from .run import Run

def convert_to_Run(mt: MeasurementTools, file=0, **kwargs):
    run = Run()
    meta = {}

    # Get file name
    if isinstance(file, str):
        fileid = mt.getFileId(self,filename)
    else:
        fileid = file
        file = mt.files[fileid][0]
    meta['filename'] = file

    # Get observable
    obs = kwargs.get('obs',0)
    if isinstance(obs, str):
        pass
    else:
        obslist = mt.extractObservables(fileid)
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
    run.bins = Run.convert_to_bins(edgesList)
    v = mt.histogramValues(hist, withOUF=withOUF)
    nsetups = v.shape[-1]
    v = v.reshape((len(run.bins),nsetups))
    e = mt.histogramErrors(hist, withOUF=withOUF).reshape((len(run.bins),nsetups))
    p = mt.histogramHits(hist, withOUF=withOUF).reshape((len(run.bins)))
    dim = len(edgesList)

    run.values = v[:,setupids]
    run.errors = e[:,setupids]
    run.xsec = np.transpose(mt.extractXSections(fileid)[setupids,:,0])

    run.dim = dim
    meta['obs'] = obs
    meta['histid'] = histid
    meta['smearing'] = mt.histogramSmearing(hist)
    meta['nevents'] = int(mt.files[fileid][1].find('nevents').text)
    run.meta.update(meta)
    run.name = file

    run.make_differential()

    return run


def load_to_Run(xmlfile, **kwargs):
    mt = MeasurementTools()
    mt.loadxml(xmlfile)
    return convert_to_Run(mt, 0, **kwargs)

