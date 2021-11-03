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
    edges:  stored as List(List(float))
    values: stored as array with (X,Y) dimensions,
            where X is the number of bins,
            and Y is the results at different scales.
    errors: stored similarly to values
    info:   other information related to the run
    """

    def __init__(self, file=None, bins=None, edges=None, nsetups=1, **kwargs):
        """Initialise either by filename and kwargs, or by specifying bins or edges"""
        if (file):
            self.load(file,**kwargs)
        else:
            if (bins):
                self.bins = bins
                self.values = np.zeros((len(bins),nsetups))
                self.errors = np.zeros((len(bins),nsetups))
            elif (edges):
                self.edges = edges
                self.values = np.zeros((len(bins),nsetups))
                self.errors = np.zeros((len(bins),nsetups))
            self.info = {}

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

    def update_info(self,info=None,**kwargs):
        """Update run information

        Optionally pass either Run or dict instance.
        Optionally pass additional flags to modify on top of that.
        """
        if isinstance(info,Run):
            self.info.update(info.info)
        elif isinstance(info,dict):
            self.info.update(info)
        self.info.update(kwargs)

    @property
    def info(self):
        if hasattr(self,'_info'):
            return self._info
        else:
            self._info = {}
            return self._info

    @info.setter
    def info(self,info):
        self._info = info

    @property
    def name(self):
        res = self.info.get('name')
        if (res == None):
            res = self.info.get('file')
        return res

    @name.setter
    def name(self, value, latex=False):
        if (latex):
            value = re.sub('_', '\\_', value)
        self.info['name'] = value

    @property
    def bins(self):
        return self._bins

    @bins.setter
    def bins(self, v):
        """Sets bins and automatically calculates corresponding edges"""
        self._bins = v
        self._edges = Run.convert_to_edges(v)

    @property
    def edges(self):
        return self._edges

    @edges.setter
    def edges(self, edges):
        """Sets edges and automatically calculates corresponding bins"""
        self._edges = edges
        self._bins = Run.convert_to_bins(self._edges)

    @property
    def shape(self):
        """Get dimensions for bins across all dimensions"""
        return [len(edges) - 1 for edges in self.edges]

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

                    edges = [[df.iat[0,1]] + list(df.iloc[:,2])]
                    bins = Run.convert_to_bins(edges)
                    vals = df.iloc[:,3:6].values
                    if (len(df.columns) == 6):
                        vals[:,1] += vals[:,0]
                        vals[:,2] += vals[:,0]
                    elif (len(df.columns) == 8):
                        syserrs = df.iloc[:,6:8].values
                        vals[:,1] = np.sqrt(vals[:,1]**2 + syserrs[:,0]**2)
                        vals[:,2] = -np.sqrt(vals[:,2]**2 + syserrs[:,1]**2)
                        vals[:,1] += vals[:,0]
                        vals[:,2] += vals[:,0]
                    else:
                        raise Exception('Supported cases: 6 or 8 columns.')

                    errs = np.zeros(vals.shape)
                    data = {'mean': [[b,v] for b,v in zip(bins,vals)],\
                            'std':  [[b,e] for b,e in zip(bins,errs)],\
                            'info': {'differential': True, 'experiment': True}}

                    data['file'] = request
                    load(self,data,**kwargs)

        return inner

    @loading_methods
    def load(self,request,**kwargs):
        """Load data to Run instance.
        Uses hightea output interface as input.
        Can be fed with dictionary or path to JSON/YAML file.
        """
        self.info = {}
        self.info['file'] = request.get('file')

        mean = request.get('mean',[])
        std = request.get('std',[])
        values = []
        errors = []
        bins = []

        for m in mean:
            bins.append(m[0])
            values.append(m[1])

        for s in std:
            errors.append(s[1])

        self.bins = bins
        self.values = np.array(values)
        self.errors = np.array(errors)

        # TODO: test
        if 'xsec' in request:
            self.xsec = np.array(request.get('xsec'))

        if 'info' in request:
            self.info.update(request.get('info'))

        # other
        # TODO: review location of this
        # self.name = request.get('name')
        # self.variable = request.get('variable')

        # Final corrections
        for key,value in kwargs.items():
            self.info[key] = value

        if not(self.is_differential()):
            self.make_differential()


    def is_differential(self):
        """Check if run set to be a differential distribution"""
        return self.info.get('differential',False)


    def make_histogramlike(self,ignorechecks=False):
        """Turn differential distribution to histogram"""
        if not(self.is_differential()) and not(ignorechecks):
            raise Exception("Already is histogram-like")
        def area(bins):
            a = 1
            for b in bins: a *= b[1]-b[0]
            return a
        areas = [ area(b) for b in self.bins ]

        for i,v in enumerate(self.values):
            self.values[i] = v*areas[i]

        for i,e in enumerate(self.errors):
            self.errors[i] = e*areas[i]

        self.info['differential'] = False
        return self


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

        self.info['differential'] = True
        return self


    def __add__(self,other):
        """Adding method"""
        res = self.minicopy()
        if (isinstance(other,Run)):
            assert(res.values.shape[0] == other.values.shape[0])
            res.values += other.values
            res.errors = np.sqrt(res.errors**2 + other.errors**2)
        elif isinstance(other,float) or isinstance(other,int):
            res.values += other
        else:
            raise Exception("Add operation failed")
        return res


    __radd__ = __add__


    def __sub__(self,other):
        """Subtraction method"""
        return self.__add__((-1.)*other)


    def __rsub__(self,other):
        """Subtraction method"""
        return other + (-1.)*self


    def __mul__(self,other):
        """Multiplication method"""
        res = self.minicopy()
        if (isinstance(other,Run)):
            assert(res.values.shape[0] == other.values.shape[0])

            res.errors = res.errors*other.values + \
                         res.values*other.errors
            res.values *= other.values

        elif isinstance(other,float) or isinstance(other,int):
            res.values *= other
            res.errors *= other
        else:
            raise Exception("Mul operation failed")
        return res


    __rmul__ = __mul__


    def __truediv__(self,other):
        """Run division method. Supports division by a constant."""
        # TODO: tackle is_differential flag consistently
        res = self.minicopy()
        warnings = np.geterr(); np.seterr(invalid='ignore')
        if (isinstance(other,Run)):
            assert(res.values.shape[0] == other.values.shape[0])

            res.errors = res.errors/other.values + \
                  res.values*other.errors/other.values**2
            res.values /= other.values

        elif isinstance(other,float) or isinstance(other,int):
            res.values /= other
            res.errors /= other
        else:
            raise Exception("Div operation failed")
        np.seterr(**warnings)
        return res


    def _get_attributes(self):
        """Get attributes from the class"""
        return [attr for attr in dir(self)
                if not(callable(getattr(self, attr)))
                and not(attr.startswith('_'))]


    def _attributes_equal(self,other,attr):
        """Check whether attribute is the same for two instances"""
        check = (getattr(self,attr) == getattr(other,attr))
        return check if (isinstance(check,bool)) else check.all()


    def __eq__(self, other):
        """Check if runs contain identical values

        All attributes are checked except additional information
        """
        members = self._get_attributes()
        other_members = other._get_attributes()
        if not(members == other_members):
            return False
        for m in members:
            if not(m == 'info'):
                if not(self._attributes_equal(other,m)):
                    return False
        return True


    def zoom(self, value=None, line=None, dim=0):
        """Zoom into one bin at one dimension to get a lower dim slice.

        Specify the bin by some value that it contains or
        directly by the line number.
        """
        if not(value == None):
            line = 0
            while (self.edges[dim][line+1] < value
                   and line < self.dimensions()[dim]-1):
                line += 1

        left,right = self.edges[dim][line:line+2]
        binpos = [i for i,x in enumerate(self.bins) if x[dim]==[left,right]]
        newrun = Run()
        newrun.values = deepcopy(self.values[binpos])
        newrun.errors = deepcopy(self.errors[binpos])
        edges = deepcopy(self.edges)
        assert(len(edges) > 1),"Zoom is intended for differential distributions with dim >= 2"
        edges.pop(dim)
        newrun.edges = edges
        newrun.info = deepcopy(self.info)
        if ('obs') in newrun.info:
            if not(value == None):
                newrun.info['obs'] += f' ({value})'
            else:
                newrun.info['obs'] += f' [line={line}]'
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

        # sync variation information with actual data
        if ('variation' in run.info):
            try:
                run.update_info(variation=self.info.get('variation')[sliced])
            except Exception:
                run.info.pop('variation')

        return run


    def minicopy(self):
        """Minimal copy: only data"""
        run = Run()
        run.bins = deepcopy(self.bins)
        run.values = deepcopy(self.values)
        run.errors = deepcopy(self.errors)
        if hasattr(self,'xsec'):
            run.xsec = deepcopy(self.xsec)
        for attr in 'experiment'.split():
            if attr in self.info:
                run.update_info(**{attr:self.info.get(attr)})
        return run


    def deepcopy(self):
        """Deepcopy the whole run data"""
        return deepcopy(self)


    def flatten(self):
        """Remove dimensions represented by single bins"""
        self.edges = [x for x in self.edges if (len(x) > 2)]


    def to_htdict(self):
        """Get dictionary in hightea format from this run"""
        res = {}
        res['mean'] = list(zip(self.bins, self.values.tolist()))
        res['std'] = list(zip(self.bins, self.errors.tolist()))
        for attr in 'xsec'.split():
            if hasattr(self,attr):
                res[attr] = getattr(self,attr)

        res['info'] = self.info
        return res


    def to_json(self,file):
        """Dump run to JSON file in hightea format"""
        with open(file, 'w') as f:
            json.dump(self.to_htdict(), f)


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
            edgesList.append(dimedges)
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
    @staticmethod
    def full(dims, scales=1, fill_value=0):
        """Get run with filled const values"""
        run = Run()
        run.edges = [list(range(d+1)) for d in dims if d > 0]
        run.values = np.full((len(run.bins),scales),float(fill_value))
        run.errors = np.full((len(run.bins),scales),0.)
        return run

    @staticmethod
    def seq(dims, scales=1):
        """Get a multi-dimensional run for testing purposes

        Fills values with sequential values.
        """
        run = Run()
        run.edges = [list(range(d+1)) for d in dims if d > 0]
        run.values = np.arange(0,len(run.bins),1./scales)\
                    .reshape(len(run.bins),scales)
        run.errors = run.values / 10
        return run

    @staticmethod
    def random(dims, scales=1):
        """Get random multi-dimensional run for testing purposes"""
        run = Run()
        run.edges = [list(range(d+1)) for d in dims if d > 0]
        run.values = np.random.rand(len(run.bins),scales)
        run.errors = np.random.rand(len(run.bins),scales) / 10
        return run


    # TODO:
    def apply(**tweaks):
        """Some specific operations to apply to run"""
        pass


    # TODO: nice printout
    def __repr__(self):
        desc = ""
        for m in self._get_attributes():
            desc += f" '{m}': {getattr(self,m)}\n"
        return desc
