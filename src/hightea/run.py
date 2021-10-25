import pathlib
import json
import re
import numpy as np
from functools import wraps
from copy import copy, deepcopy

class Run(object):
    """
    Run encapsulates histogram data for an observable and its metadata

    bins:   stored as List(List(List(float)))
            for example: [ [[0,1],[2,3]], [[1,2],[2,3] ] ]
                represents 2d observable with edges [0,1,2], [2,3]
    edges:  stored as List(array(float))
    values: stored as array with (X,Y) dimensions,
            where X is the number of bins,
            and Y is the results at different scales.
    errors: stored similarly to values
    meta:   other information related to the run
    """
    # TODO: consider using setters and getters instead of accessing objects directly

    def __init__(self, bins=None, edges=None, name=None, nsetups=1):
        self.initialised = False
        self.meta = {}
        self._isdifferential = False
        if (bins):
            self.bins = bins
            self.edges = Run.convert_to_edges(self.bins)
            self.values = np.zeros((len(bins),nsetups))
            self.errors = np.zeros((len(bins),nsetups))
        elif (edges):
            self.edges = edges
            self.bins = Run.convert_to_bins(self.edges)
            self.values = np.zeros((len(bins),nsetups))
            self.errors = np.zeros((len(bins),nsetups))
        if (name):
            self.meta['name'] = name
        pass

    @property
    def name(self):
        return self.meta.get('name')

    @name.setter
    def name(self, value, latex=False):
        if (latex):
            value = re.sub('_', '\\_', value)
        self.meta['name'] = value

    def v(self):
        """Get values at central scale"""
        return self.values[:,0]

    def e(self):
        """Get errors at central scale"""
        return self.errors[:,0]

    def upper(self):
        """Get upper values for scale variation"""
        return np.amax(self.values, axis=1)

    def lower(self):
        """Get lower values for scale variation"""
        return np.amin(self.values, axis=1)

    def dim(self):
        """Get dimension of the run"""
        return len(self.edges)

    def update_meta(**info):
        """Update run meta information"""
        self.meta.update(info)

    def loading_methods(load):
        @wraps(load)
        def inner(self,request):
            if (isinstance(request,dict)):
                load(self,request)
            elif (isinstance(request,str)):
                ext = pathlib.Path(request).suffix
                if (ext) == '.json':
                    with open(request,'r') as f:
                        data = json.load(f)
                    data['file'] = request
                    load(self,data)
        return inner

    @loading_methods
    def load(self,request):
        """Load data to Run instance.
        Uses hightea output interface as input.
        Can be fed with dictionary or path to JSON/YAML file.
        """
        self.meta['file'] = request.get('filename')

        mean = request.get('mean',[])
        std = request.get('std',[])
        self.values = []
        self.errors = []
        self.bins = []

        for m in mean:
            self.bins.append(m[0])
            self.values.append(m[1])

        for s in std:
            self.errors.append(s[1])

        self.values = np.array(self.values)
        self.errors = np.array(self.errors)
        self.edges = self.convert_to_edges(self.bins)
        self.make_differential()

        if 'xsec' in request:
            self.xsec = np.array(request.get('xsec'))

        # other
        self.name = request.get('name')
        self.variable = request.get('variable')


    def is_differential(self):
        return self._isdifferential


    def make_differential(self):
        """Turn histograms into differential distributions"""
        if (self.is_differential()):
            raise Exception("Already made differential")
        def area(bins):
            a = 1
            for b in bins: a *= b[1]-b[0]
            return a
        areas = [ area(b) for b in self.bins ]

        for i,v in enumerate(self.values):
            self.values[i] = v/areas[i]

        for i,e in enumerate(self.errors):
            self.errors[i] = e/areas[i]

        self._isdifferential = True
        return self


    def __mul__(self,other):
        """Multiplication method"""
        res = self.mincopy()
        if (isinstance(other,Run)):
            assert(res.values.shape[0] == other.values.shape[0])
            if (other.values.shape[1] == 1):
                s = np.newaxis
            elif (res.values.shape[1] == other.values.shape[1]):
                s = slice(None)

            res.values *= other.values[:,s]
            res.errors = res.errors*other.values[:,s] + \
                         res.values*other.errors[:,s]

        elif (isinstance(denom,float)):
            res.values *= other
            res.errors *= other
        else:
            raise Exception("Mul operation failed")
        return res


    # TODO: test division examples
    def __truediv__(self,other):
        """Run division method. Supports division by a constant."""
        res = self.mincopy()
        if (isinstance(other,Run)):
            assert(res.values.shape[0] == other.values.shape[0])
            if (other.values.shape[1] == 1):
                s = np.newaxis
            elif (res.values.shape[1] == other.values.shape[1]):
                s = slice(None)

            res.values /= other.values[:,s]
            res.errors = res.errors/other.values[:,s] + \
                  res.values*other.errors[:,s]/other.values[:,s]**2

        elif (isinstance(denom,float)):
            res.values /= other
            res.errors /= other
        else:
            raise Exception("Div operation failed")
        return res


    # TODO: generalise
    def get_1d_slice(self, line):
        """Get a slice from 2D observable for 1D distribution"""
        assert len(self.edges)==2, "get_1d_slice only works with 2D runs"

        left,right = self.edges[0][line:line+2]
        binpos = [i for i,x in enumerate(self.bins) if x[0]==[left,right]]
        newrun = Run()
        newrun.values = deepcopy(self.values[binpos])
        newrun.errors = deepcopy(self.errors[binpos])
        newrun.edges = [deepcopy(self.edges[1])]
        newrun.bins = self.convert_to_bins(newrun.edges)
        newrun.meta = deepcopy(self.meta)
        newrun.meta['obs'] += f' [{line}]'
        return newrun


    def __getitem__(self,sliced):
        """Get a subset of the full run (operating with dimensional bins)"""

        # this is to use edges as dimensional bins
        def increase_slice_stop(s):
            return s if (s.stop == None) else slice(s.start, s.stop+1, s.step)

        edges = deepcopy(self.edges)
        if isinstance(sliced,list):
            for i,s in enumerate(sliced):
                edges[i] = edges[i][increase_slice_stop(s)]
        elif isinstance(sliced,slice):
            edges[0] = edges[0][increase_slice_stop(sliced)]
        else:
            sliced = slice(sliced, sliced+2, None)
            edges[0] = edges[0][sliced]

        bins = Run.convert_to_bins(edges)
        run = Run(bins)

        selection = [b in run.bins for b in self.bins]
        run.values = deepcopy(self.values[selection,:])
        run.errors = deepcopy(self.errors[selection,:])

        for attr in 'obs smearing nevents histid'.split():
            if (attr in self.meta):
                run.meta[attr] = deepcopy(self.meta[attr])
        return run

    def mincopy(self):
        """Minimal copy: only data"""
        run = Run()
        run.bins = deepcopy(self.bins)
        run.edges = Run.convert_to_edges(run.bins)
        run.values = deepcopy(self.values)
        run.errors = deepcopy(self.errors)
        if hasattr(self,'xsec'):
            run.xsec = deepcopy(self.xsec)
        return run

    def deepcopy(self):
        """Deepcopy the whole run data"""
        return deepcopy(self)

    def flatten(self):
        """Remove dimensions represented by one bin"""
        self.edges = [x for x in self.edges if (len(x) > 2)]
        self.bins = Run.convert_to_bins(self.edges)


    @staticmethod
    def convert_to_edges(binsList):
        """Get edges for each dimension given a list of bins"""
        if len(binsList[0]) == 1:
            return [np.array([ binsList[0][0][0] ]
                           + [ bins[0][1] for bins in binsList ])]
        ndims = len(binsList[0])
        edgesList = []
        for dim in range(0, ndims):
            dimedges = [binsList[0][dim][0]]
            for i,bins in enumerate(binsList):
                if not(bins[dim][1] in dimedges):
                    dimedges.append(bins[dim][1])
                else:
                    if len(dimedges)>2 and bins[dim][0] == dimedges[0]:
                        break
            edgesList.append(np.array(dimedges))
        return edgesList


    @staticmethod
    def convert_to_bins(edgesList):
        """Get full list of bins given edges for each dimension"""
        edges = edgesList[-1]
        if (len(edgesList) == 1):
            return [ [[a,b]] for a,b in zip(edges[:-1],edges[1:]) ]
        else:
            shortbinsList = Run.convert_to_bins(edgesList[:-1])
            binsList = []
            for bins in shortbinsList:
                for newbin in Run.convert_to_bins([edges]):
                    binsList.append(bins + newbin)
            return binsList


    @staticmethod
    def random(dims, scales=1):
        """Get random multi-dimensional run for testing purposes"""
        run = Run()
        run.edges = [np.array(range(d+1)) for d in dims if d > 0]
        run.bins = Run.convert_to_bins(run.edges)
        run.values = np.random.rand(len(run.bins),scales)
        run.errors = np.random.rand(len(run.bins),scales) / 10
        return run


    # TODO:
    def apply(**tweaks):
        """Some specific operations to apply to run"""
        pass

    # TODO: nice printout
    def __repr__(self):
        return self.name
