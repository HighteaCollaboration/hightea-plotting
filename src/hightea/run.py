import pathlib
import json
import re
import numpy as np
import pandas as pd
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

    def __init__(self, file=None, bins=None, edges=None, nsetups=1, **kwargs):
        """Initialise either by filename and kwargs, or by specifying bins or edges"""
        if (file):
            self.load(file,**kwargs)
        else:
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

    @property
    def meta(self):
        if hasattr(self,'_meta'):
            return self._meta
        else:
            self._meta = {}
            return self._meta

    @meta.setter
    def meta(self,meta):
        self._meta = meta

    @property
    def name(self):
        res = self.meta.get('name')
        if (res == None):
            res = self.meta.get('file')
        return res

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

    def dimensions(self):
        """Get dimensions for each axis"""
        return [len(x)-1 for x in self.edges]

    def nsetups(self):
        """Get number of setups in run"""
        return self.values.shape[1]

    def update_meta(self,**info):
        """Update run meta information"""
        self.meta.update(info)

    # TODO: debug and use set/getters below
    # @property
    # def bins(self):
    #     return self._bins
    #
    # @bins.setter
    # def bins(self, bins):
    #     self._bins = bins
    #     self._edges = Run.convert_to_edges(self._bins)
    #
    # @property
    # def edges(self):
    #     return self._edges
    #
    # @edges.setter
    # def edges(self, edges):
    #     self._edges = edges
    #     self._bins = Run.convert_to_bins(self._edges)


    # TODO: loading from HEPDATA file
    def loading_methods(load):
        @wraps(load)
        def inner(self,request,**kwargs):
            if (isinstance(request,dict)):
                load(self,request,**kwargs)
            elif (isinstance(request,str)):
                ext = pathlib.Path(request).suffix

                if (ext) == '.json':
                    """File format as provided by hightea"""
                    with open(request,'r') as f:
                        data = json.load(f)
                    data['file'] = request
                    load(self,data,**kwargs)

                if (ext) == '.csv':
                    """File format as provided by HEPDATA"""
                    df = pd.read_csv(request,comment='#')

                    if not(len(df.columns) == 6):
                        # TODO: generalise
                        raise Exception('Expecting 6 columns, other cases not implemented yet')

                    edges = [np.array([df.iat[0,1]] + list(df.iloc[:,2]))]
                    bins = Run.convert_to_bins(edges)
                    vals = df.iloc[:,3:6].values
                    vals[:,1] += vals[:,0]
                    vals[:,2] += vals[:,0]
                    errs = np.zeros(vals.shape)
                    data = {'mean': [[b,v] for b,v in zip(bins,vals)],\
                            'std':  [[b,e] for b,e in zip(bins,errs)]}

                    data['file'] = request
                    load(self,data,**kwargs)

        return inner

    @loading_methods
    def load(self,request,**kwargs):
        """Load data to Run instance.
        Uses hightea output interface as input.
        Can be fed with dictionary or path to JSON/YAML file.
        """
        self.meta = {}
        self.meta['file'] = request.get('file')

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

        # if 'xsec experiment' in request:
        #     self.xsec = np.array(request.get('xsec'))

        # other
        self.name = request.get('name')
        self.variable = request.get('variable')

        # Final corrections
        for key,value in kwargs.items():
            self.meta[key] = value


    def is_differential(self):
        """Check if run set to be a differential distribution"""
        if hasattr(self,'_is_differential'):
            return self._is_differential
        else:
            return False


    def make_differential(self):
        """Turn histograms into differential distributions"""
        if self.is_differential():
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

        self._is_differential = True
        return self


    # TODO: test multiplication examples
    def __mul__(self,other):
        """Multiplication method"""
        res = self.minicopy()
        if (isinstance(other,Run)):
            assert(res.values.shape[0] == other.values.shape[0])

            res.values *= other.values
            res.errors = res.errors*other.values + \
                         res.values*other.errors

        elif (isinstance(denom,float)):
            res.values *= other
            res.errors *= other
        else:
            raise Exception("Mul operation failed")
        return res


    # TODO: test division examples
    def __truediv__(self,other):
        """Run division method. Supports division by a constant."""
        res = self.minicopy()
        if (isinstance(other,Run)):
            assert(res.values.shape[0] == other.values.shape[0])

            warnings = np.geterr()
            np.seterr(invalid='ignore')

            res.values /= other.values
            res.errors = res.errors/other.values + \
                  res.values*other.errors/other.values**2

            np.seterr(**warnings)

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


    # @property
    # def xsec(self):
    #     if hasattr(self,'_xsec'):
    #         return self._xsec
    #     else:
    #         # TODO: log warning here
    #         return None
    #
    # @xsec.setter
    # def xsec(self,v):
    #     assert(len(v.shape) == 2)
    #     self._xsec = v


    def __getitem__(self,sliced):
        """Get a run with selected setups"""
        if isinstance(sliced,list):
            raise Exception('List not expected')
        elif isinstance(sliced,int):
            sliced = slice(sliced,sliced+1)

        run = deepcopy(self)
        for a in 'values errors xsec'.split():
            if hasattr(run,a):
                setattr(run,a,getattr(self,a)[:,sliced])
        return run


    def minicopy(self):
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
        """Remove dimensions represented by single bins"""
        self.edges = [x for x in self.edges if (len(x) > 2)]
        self.bins = Run.convert_to_bins(self.edges)


    # TODO: define
    def to_json(self):
        pass


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

    # # TODO: test
    # @staticmethod
    # def full(dims, fill_value=0, scales=1):
    #     """Get run with filled const values"""
    #     run = Run()
    #     run.edges = [np.array(range(d+1)) for d in dims if d > 0]
    #     run.bins = Run.convert_to_bins(run.edges)
    #     run.values = np.full(len(run.bins),scales)
    #     run.errors = np.random.rand(len(run.bins),scales) / 10

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
