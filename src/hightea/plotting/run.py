from pathlib2 import Path
import json
import re
import warnings
import numpy as np
import pandas as pd
from functools import wraps
from copy import copy, deepcopy

class Run(object):
    """Container for observable results and metadata

    Designed to be a universal container for histogram
    and differential distribution results from hightea,
    HEPDATA, and other sources.
    Stores values and statistical errors as simple
    numpy arrays, with two dimensions, corresponding
    to the list of bins, and the calculation setups.

    Attributes
    ----------
    values: numpy.ndarray
        Values for histogram or differential distribution
        across all bins and scale setups.
        Has (X,Y) shape, where X is the number of bins,
        and Y is the number of scale setups.

    errors: numpy.ndarray
        Statistical errors corresponding to values.
        Same shape as values.
    """

    def __init__(self, file=None, bins=None, edges=None, nhist=0,
                nsetups=1, **kwargs):
        """Run class constructor.

        Creates and returns and instance of Run class based on provided
        information. To parse a file or a constructed object, use `file`
        parameters, otherwise, specify either bins or edges.

        Parameters
        ----------
        optional file : str, dict, array
            Path to file or an instance of an object.
            Several formats supported: JSON, CSV, dictionary.

        optional bins : list
            Bins in the format of 3D-list: [... [ [l1,r1], [l2,r2], ...], ...]

        optional edges : list
            Bin edges in the format of 2D-list: [ ... , [x0, x1 ... ], ... ]

        Returns
        -------
        Run
            Instance of Run class.

        Examples
        --------
        >>> run = Run('req.json')
        >>> run = Run('experiment.csv')
        >>> run = Run(dict(...))
        >>> run = Run(edges=[[0,1,2,3,4,5],[-1,-0.5,0,0.5,1]])
        """
        if (file):
            self.load(file,nhist=nhist,**kwargs)
        else:
            if (bins):
                self.bins = bins
                self.values = np.zeros((len(bins),nsetups))
                self.errors = np.zeros((len(bins),nsetups))
            elif (edges):
                self.edges = edges
                self.values = np.zeros((len(self.bins),nsetups))
                self.errors = np.zeros((len(self.bins),nsetups))
            self.info = {}

    def v(self):
        """Get values at central scale

        Returns
        -------
        numpy.ndarray
            Array of central scale values for each bin correspondingly.
        """
        return self.values[:,0]

    def e(self):
        """Get errors at central scale

        Returns
        -------
        numpy.ndarray
            Array of central scale errors for each bin correspondingly.
        """
        return self.errors[:,0]

    def upper(self):
        """Get upper values for scale variation

        Returns
        -------
        numpy.ndarray
            Array of upper scale band values for each bin correspondingly.
        """
        return np.amax(self.values, axis=1)

    def lower(self):
        """Get lower values for scale variation

        Returns
        -------
        numpy.ndarray
            Array of upper scale band values for each bin correspondingly.
        """
        return np.amin(self.values, axis=1)

    def dim(self):
        """Get dimension of the run

        Returns
        -------
        int
            Number of dimensions in data.
        """
        return len(self.edges)

    def dimensions(self):
        """Get dimensions for each axis

        Returns
        -------
        list
            List of bin numbers for each dimension.
        """
        return [len(x)-1 for x in self.edges]

    def nsetups(self):
        """Get number of setups in run

        Returns
        -------
        int
            Number of different scale setups in the run.
        """
        return self.values.shape[1]

    def update_info(self,info=None,**kwargs):
        """Update run information

        Parameters
        ----------
        optional info:
            - Run: copy metadata from passed run
            - str: set the name of the run
            - dict: update metadata with dictionary

        **kwargs
            Arbitrary information will be passed on to run.info dictionary.

        Returns
        -------
        None
        """
        if isinstance(info,Run):
            self.info.update(info.info)
        elif isinstance(info,str):
            self.name = info
        elif isinstance(info,dict):
            self.info.update(info)
        self.info.update(kwargs)
        return self

    @property
    def info(self):
        """Retrieve run metadata

        Returns
        -------
        dict
        """
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
        """Retrieve run name

        Returns
        -------
        str
        """
        res = self.info.get('name')
        if (res == None):
            res = self.info.get('file','')
        return res

    @name.setter
    def name(self, value, latex=False):
        if (latex):
            value = re.sub('_', '\\_', value)
        self.info['name'] = value

    @property
    def bins(self):
        """Retrieve run bins

        Returns
        -------
        list
        """
        return self._bins

    @bins.setter
    def bins(self, v):
        """Sets bins and automatically calculates corresponding edges"""
        self._bins = v
        self._edges = Run.convert_to_edges(v)

    @property
    def edges(self):
        """Retrieve run edges

        Returns
        -------
        list
        """
        return self._edges

    @edges.setter
    def edges(self, edges):
        """Sets edges and automatically calculates corresponding bins"""
        self._edges = edges
        self._bins = Run.convert_to_bins(self._edges)

    @staticmethod
    def bin_area(bins):
        """Calculate area for a multidimensional histogram bin.

        Returns
        -------
        float
        """
        a = 1
        for b in bins: a *= b[1]-b[0]
        return a

    def loading_methods(load):
        """Convenient loading methods to load data into Run class"""
        @wraps(load)
        def inner(self,request,**kwargs):
            if (isinstance(request,dict)):
                load(self,request,**kwargs)
            else:
                # assumes path and coverts to string
                request = str(request)
                ext = '.'+kwargs.get('ext',Path(request).suffix[1:])

                if (ext) == '.json':
                    """File format as provided by hightea"""
                    with open(request,'r') as f:
                        data = json.load(f)
                    data['file'] = request
                    load(self,data,**kwargs)

                elif (ext) == '.csv':
                    """File format as provided by HEPDATA"""
                    header = kwargs.get('header','infer')
                    df = pd.read_csv(request,header=header,comment='#')

                    edges = [[df.iat[0,1]] + list(df.iloc[:,2])]
                    bins = Run.convert_to_bins(edges)
                    if (len(df.columns) < 6):
                        raise Exception(f'Bad number of columns in CSV file')
                    elif len(df.columns) == 6:
                        vals = df.iloc[:,3:6].values
                        vals[:,1] += vals[:,0]
                        vals[:,2] += vals[:,0]
                        errs = np.zeros(vals.shape)
                    elif len(df.columns) > 6:
                        assert len(df.columns) > 7, 'Bad number of columns in CSV file'
                        if ((len(df.columns) > 7) and \
                            ((df.columns[6] <= 0).all() or (df.columns[7] <= 0).all())):
                            # interpret as combination of sys errors
                            vals = df.iloc[:,[3,6,7]].values
                            for i in range(8, len(df.columns)):
                                vals[:,1:] += df.iloc[:,[i,i+1]].values
                            vals[:,1] += vals[:,0]
                            vals[:,2] += vals[:,0]
                            errs = np.zeros(vals.shape)
                            errs[:,0] = (df.iloc[:,4].values - df.iloc[:,5].values)/2
                            errs[:,1] = errs[:,0]
                            errs[:,2] = errs[:,0]
                        else:
                            vals = df.iloc[:,6:].values
                            errs = np.zeros(vals.shape)
                            errs[:,0] = (df.iloc[:,4].values - df.iloc[:,5].values)/2
                            errs[:,1] = errs[:,0]
                            errs[:,2] = errs[:,0]
                    else:
                        raise Exception(f'Meaning of {len(df.columns)} columns in CSV file is unclear.')

                    data = {'histogram': [
                              {
                                'edges': [{
                                            'min_value':b[0],
                                            'max_value':b[1],
                                          }
                                          for i,b in enumerate(bb)],
                                'mean': v,
                                'error': e,
                              }
                              for bb,v,e in zip(bins,vals,errs)
                            ]}
                    data['info'] = {'file': request,
                                    'differential': True,
                                    'experiment': True}
                    load(self,data,**kwargs)

                else:
                    raise Exception(f'Unexpected input format: {ext}')

        return inner

    @loading_methods
    def load(self,request,nhist=0,**kwargs):
        """Load data to Run.

        Parameters
        ----------
        request : dict
            Python dictionary in the JSON format as returned by hightea server.

        **kwargs
            Any modifications to the metadata dictionary.

        Returns
        -------
        None
        """
        hist = request.get('histogram')
        if not hist:
            try:
                hist = request.get('histograms')[nhist]
            except IndexError as e:
              print(f'Histogram #{nhist} not found', e)

        assert len(hist) > 0, "Histogram is empty"
        bins = []
        values = []
        errors = []
        syserrs = []
        for entry in hist:
            bins.append(entry.get('edges',[[]]))
            values.append(entry.get('mean',[]))
            errors.append(entry.get('error',[]))
            syserrs.append(entry.get('sys_error',[]))

        # two possible bin formats
        if isinstance(bins[0][0], dict):
            bins = [[[b['min_value'],b['max_value']] for b in bb] for bb in bins]
        self.bins = bins

        # values & errors can be 1d or 2d lists
        if (isinstance(values[0],list) or isinstance(values[0],np.ndarray)):
            self.values = np.array(values)
            self.errors = np.array(errors)
            if len(syserrs[0]) > 0:
                warnings.warn("sys_errors are ignored since values are passed as arrays")
        else:
            # create 3-setup run if sys-errors are passed
            if len(syserrs[0]) > 0:
                pos_edge = np.zeros(len(values))
                neg_edge = np.zeros(len(values))
                for i, (v,syserr) in enumerate(zip(values, syserrs)):
                    neg_edge[i] = v - np.sqrt(sum([e.get('neg',0)**2 for e in syserr]))
                    pos_edge[i] = v + np.sqrt(sum([e.get('pos',0)**2 for e in syserr]))

                self.values = np.transpose(np.array([np.array(values)]
                                                  + [neg_edge]
                                                  + [pos_edge]))
                self.errors = np.transpose(np.array(3*[errors]))
            else:
                self.values = np.expand_dims(np.array(values),1)
                self.errors = np.expand_dims(np.array(errors),1)

        # retrieve info
        self.info = dict(request.get('info',{}))
        self.info['file'] = request.get('file')

        # manipulating data from original request
        if 'request' in self.info:
            req = self.info.get('request')
            if type(req) == str:
                req = json.loads(req)
                self.info['request'] = req
            if 'observables' in req:
                obslist = req.get('observables')
                try:
                    self.info['obs'] = ' * '.join([x.get('variable') for x in obslist[nhist]])
                except IndexError:
                    warnings.warn("Error when retrieving observable name")

        if 'fiducial_mean' in request:
            xsec = [request.get('fiducial_mean')]
            if not(isinstance(xsec[0],list)): xsec[0] = [xsec[0]]
            if 'fiducial_sys_error' in request:
                syserr = request.get('fiducial_sys_error')
                xsec[0].append(xsec[0][0] - np.sqrt(sum([e.get('neg',0)**2 for e in syserr])))
                xsec[0].append(xsec[0][0] + np.sqrt(sum([e.get('pos',0)**2 for e in syserr])))

            xsec.append(request.get('fiducial_error', [0.]*len(xsec[0])))
            if not(isinstance(xsec[1],list)): xsec[1] = [xsec[1]]
            if len(xsec[0]) > 1 and len(xsec[1]) == 1: xsec[1] *= len(xsec[0])

            self.xsec = np.array(xsec,dtype=float)

        # Final corrections
        for key,value in kwargs.items():
            self.info[key] = value

        if not(self.is_differential()):
            self.make_differential()


    def is_differential(self):
        """Check if run set to be a differential distribution

        Looks into self.info['differential'] to see how whether run
        is set to be a histogram or differential distribution.
        By default, runs are treated as histograms.

        Returns
        -------
        bool
        """
        return self.info.get('differential',False)


    def make_histogramlike(self,ignorechecks=False):
        """Turn differential distribution to histogram

        Checks data is already set to be a histogram, and
        performs normalisation over bin area if not.

        Returns
        -------
        Run
            Modifies and returns self.
        """
        if not(self.is_differential()) and not(ignorechecks):
            warnings.warn("Already is histogram-like")
            return self
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
        """Turn histograms into differential distributions

        Checks data is already set to be a differential distribution, and
        performs normalisation over bin area if not.

        Returns
        -------
        Run
            Modifies and returns self.
        """
        if self.is_differential():
            warnings.warn("Already is differential")
            return self

        self.remove_OUF(inplace=True)

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


    def __add__(self,other):
        """Add runs.

        Sum of two runs.
        Requires dimensions to match exactly.
        Errors are propagated. Metadata is discarded.

        Returns
        -------
        Run
        """
        res = self.minicopy()
        if (isinstance(other,Run)):
            len_self, len_other = res.values.shape[0], other.values.shape[0]
            if not(len_self == len_other):
                raise Exception(f"Incompatible run shapes: {len_self}, {len_other}")
            res.values += other.values
            res.errors = np.sqrt(res.errors**2 + other.errors**2)

            if hasattr(self,'xsec') and hasattr(other, 'xsec'):
                res.xsec = self.xsec.copy()
                res.xsec[0,:] += other.xsec[0,:]
                res.xsec[1,:] = np.sqrt(res.xsec[1,:]**2 + other.xsec[1,:]**2)

        elif isinstance(other,float) or isinstance(other,int):
            res.values += other
        else:
            raise Exception("Add operation failed")
        return res


    __radd__ = __add__


    def __sub__(self,other):
        """Subtract runs.

        Sum of two runs.
        Requires dimensions to match exactly.
        Errors are propagated. Metadata is discarded.

        Returns
        -------
        Run
        """
        return self.__add__((-1.)*other)


    def __rsub__(self,other):
        return other + (-1.)*self


    def __mul__(self,other):
        """Product of run and another run|numpy.ndarray|float bin-by-bin.

        Product of run and another run|numpy.ndarray|float bin-by-bin.
        Requires dimensions to match exactly.
        Errors are propagated. Metadata is discarded.

        Parameters
        ----------
        other:
            - Run: multily bin-by-bin.
            - numpy.ndarray: multily bin-by-bin.
            - float: multiply by an overall coefficient.

        Returns
        -------
        Run
        """
        res = self.minicopy()
        if (isinstance(other,Run)):
            assert(res.values.shape[0] == other.values.shape[0])

            res.errors = res.errors*other.values + \
                         res.values*other.errors
            res.values *= other.values

        elif isinstance(other,float) or isinstance(other,int):
            res.values *= other
            res.errors *= other

        elif isinstance(other,np.ndarray):
            if len(res.bins) == other.shape[0]: # multiply binwise by array
                if len(other.shape) == 1:
                    res.values *= other[:,np.newaxis]
                    res.errors *= other[:,np.newaxis]
                else:
                    res.values *= other
                    res.errors *= other
            else:
                raise Exception(f"ndarray shape: {other.shape} "
                                + "incompatible to run {self.dimensions}")
        else:
            raise Exception("Mul operation failed")
        return res


    __rmul__ = __mul__


    def __truediv__(self,other):
        """Division of run over another run|numpy.ndarray|float bin-by-bin.

        Division of run over another run|numpy.ndarray|float bin-by-bin.
        Requires dimensions to match exactly.
        Errors are propagated. Metadata is discarded.

        Parameters
        ----------
        other:
            - Run: divide bin-by-bin.
            - numpy.ndarray: divide bin-by-bin.
            - float: divide by an overall coefficient.

        Returns
        -------
        Run
        """
        # TODO: tackle is_differential flag consistently
        res = self.minicopy()
        warnings = np.geterr(); np.seterr(invalid='ignore')
        if (isinstance(other,Run)):
            len_self, len_other = res.values.shape[0], other.values.shape[0]
            if not(len_self == len_other):
                raise Exception(f"Incompatible run shapes: {len_self}, {len_other}")

            res.errors = np.abs(res.errors/other.values + \
                  res.values*other.errors/other.values**2)
            res.values /= other.values

        elif isinstance(other,float) or isinstance(other,int):
            res.values /= other
            res.errors /= np.abs(other)

        elif isinstance(other,np.ndarray):
            if len(res.bins) == other.shape[0]: # divide binwise by array
                if len(other.shape) == 1:
                    res.values /= other[:,np.newaxis]
                    res.errors /= other[:,np.newaxis]
                else:
                    res.values /= other
                    res.errors /= other
            else:
                raise Exception(f"ndarray shape: {other.shape} "
                                + "incompatible to run {self.dimensions}")
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
        """Check if runs are equal.

        True if all non-callable attrubutes of the class except for 'info'
        are identical, otherwise False.

        Returns
        -------
        bool
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


    def abs(self):
        """Return run with absolute values

        Creates a deep copy of self and modifies run name.

        Returns
        -------
        Run
        """
        run = self.deepcopy()
        run.values = np.abs(run.values)
        if 'name' in run.info:
            run.update_info(name=run.name+' (abs)')
        return run


    def has_OUF(self):
        """Check if contains over- and underflow bins.

        Returns
        -------
        bool
        """
        for d in range(self.dim()):
            if float('inf') in [abs(x) for x in self.edges[d]]:
                return True
        return False


    def remove_OUF(self,inplace=False):
        """Remove over- and underflow bins

        Returns run without over- and underflow bins given self.

        Parameters
        ----------
        optional inplace : bool
            If True, modify in-place and return self.

        Returns
        -------
        Run
        """
        run = self if inplace else self.deepcopy()
        if self.has_OUF():
            poslist = [i for i,bins in enumerate(run.bins)
                if not float('inf') in (abs(e) for edges in bins for e in edges)]
            run.bins = [run.bins[i] for i in poslist]
            run.values = run.values[poslist]
            run.errors = run.errors[poslist]
        return run


    def zoom(self, value=None, line=None, dim=0):
        """Get run with lower dimensional slice of the data.

        Specify the bin by some value that it contains or
        directly by the line number.
        The metadata is passed on as is, with modified observable name.

        Parameters
        ----------
        optional value : float
            The slice will contain provided value.

        optional line : int
            The slice will be taken at this bin number.

        optional dim : int
            Dimension at which to take the slice.

        Returns
        -------
        Run
        """
        if not(value == None):
            line = 0
            while (self.edges[dim][line+1] <= value
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


    def mergebins(self, values=None, pos=None):
        """Merge bins by values or positions

        Specify the values or positions for bins to be
        merged into one.
        The metadata is passed on as is.
        Only 1-dim runs are supported.

        Parameters
        ----------
        optional value : list
            List with 2 values [l,r] is expected.
            bins [a,b] which satisfy l <= a, b < r will be
            merged into one bin.

        optional pos : list
            List with 2 values [l,r] is expected.
            bins with id: l <= id < r (inclusively) will be
            merged into one bin.

        Returns
        -------
        Run
        """
        # TODO: support multidimensional run

        assert self.dim() == 1,\
                "mergebins only accepts 1-dim runs"
        dim = 0
        newrun = self.minicopy(copyinfo=True)

        if (values):
            assert len(values) == 2
            l, r = values
            edges = np.array(newrun.edges[dim])
            bins_to_merge = [(i,bb) for i,bb in enumerate(self.bins) \
                        if bb[dim][0] >= l and bb[dim][0] < r]

        elif (pos):
            assert len(pos) == 2
            l, r = pos
            edges = np.array(newrun.edges[dim])
            bins_to_merge = [(i,bb) for i,bb in enumerate(self.bins) \
                        if i >= l and i < r]    # not including the right bin

        else:
            raise Exception("Bad input to mergebins")

        bins_to_merge = sorted(bins_to_merge, reverse=True, key=lambda x:x[0])
        if (len(bins_to_merge) < 2):
            return newrun

        merged_values = np.zeros((self.nsetups()))
        merged_sqerrs = np.zeros((self.nsetups()))
        new_bins = list(self.bins)
        new_values = list(self.values)
        new_errors = list(self.errors)

        if self.is_differential():
            factor = np.array([self.bin_area(b) for _,b in bins_to_merge])
        else:
            factor = np.ones((len(bins_to_merge)))

        for i,b in enumerate(bins_to_merge):
            bid = b[0]
            merged_values += new_values.pop(bid) * factor[i]
            merged_sqerrs += (new_errors.pop(bid) * factor[i])**2
            new_bins.pop(bid)

        if self.is_differential():
            total_area = np.sum(factor)
            merged_values /= total_area
            merged_sqerrs /= total_area**2

        i,_ = bins_to_merge[-1]
        new_values.insert(i, merged_values)
        new_errors.insert(i, np.sqrt(merged_sqerrs))
        new_bins.insert(i, [[bins_to_merge[-1][1][0][0],\
                             bins_to_merge[0][1][0][1]]])

        newrun = Run(bins=new_bins)
        newrun.values = np.array(new_values)
        newrun.errors = np.array(new_errors)
        newrun.info = dict(self.info)

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
        """Get a run with selected setups

        Specify slice or int to return a run with selected setups.

        Parameters
        ----------
        sliced : int, slice

        Returns
        -------
        Run
        """
        if isinstance(sliced,list):
            raise Exception('List not expected')
        elif isinstance(sliced,int):
            sliced = slice(sliced,sliced+1)

        run = deepcopy(self)
        for a in 'values errors xsec'.split():
            if hasattr(run,a):
                setattr(run,a,getattr(self,a)[:,sliced])

        # sync variation information with actual data
        if ('variation' in self.info):
            variation = deepcopy(self.info['variation'])
            if (type(variation) == list and len(variation) == self.nsetups()):
                run.update_info(variation=variation[sliced])
            else:
                del run.info['variation']
                warnings.warn(f'info.variation dropped due to mismatch with data.')

        return run


    def minicopy(self, copyinfo=False):
        """Minimal copy of run

        Only data.

        Parameters
        ----------
        optional copyinfo : bool
            If True, will include metadata.

        Returns
        -------
        Run
        """
        run = Run()
        run.bins = deepcopy(self.bins)
        run.values = deepcopy(self.values)
        run.errors = deepcopy(self.errors)
        if hasattr(self,'xsec'):
            run.xsec = deepcopy(self.xsec)
        for attr in 'experiment'.split():
            if attr in self.info:
                run.update_info(**{attr:self.info.get(attr)})
        if copyinfo:
            run.info = dict(self.info)
        return run


    def deepcopy(self):
        """Full (deep) copy of run

        Returns
        -------
        Run
        """
        return deepcopy(self)


    def flatten(self):
        """Remove dimensions represented by single bins

        Returns
        -------
        Run
        """
        self.edges = [x for x in self.edges if (len(x) > 2)]


    def to_htdict(self):
        """Get dictionary in hightea format from run

        Returns
        -------
        dict
            Dictionary in hightea format.
        """
        res = {}
        values = self.values.tolist()
        errors = self.errors.tolist()
        res['histogram'] = [{
                            'edges': [{
                                        'min_value':b[0],
                                        'max_value':b[1],
                                      }
                                      for b in bb],
                            'mean': v[0] if len(v) == 1 else v,
                            'error': e[0] if len(e) == 1 else e,
                            }
                            for bb,v,e in zip(self.bins, values, errors)
        ]
        if hasattr(self,'xsec'):
            res['fiducial_mean'] = self.xsec[:,0]
            res['fiducial_error'] = self.xsec[:,1]

        res['info'] = self.info
        return res


    def to_json(self,file,combined=False,verbose=True):
        """Dump run to JSON file in hightea format

        Parameters
        ----------
        file : str
            Output file.

        combined : bool, default: False
            If true, will print all setups into one file.
            Otherwise, will print each setup separately into different files.

        verbose : bool, default: True
            If true, will print all setups into one file.
            Otherwise, will print each setup separately into different files.

        Returns
        -------
        None
        """
        if combined:
            with open(file, 'w') as f:
                json.dump(self.to_htdict(), f)
                if verbose:
                    print(f'Saved to "{file}"')
        else:
            basefile = Path(file)
            for i in range(self.nsetups()):
                numbered_file = str(basefile.parent / basefile.stem) \
                                + f'-{i}{basefile.suffix}'
                with open(numbered_file, 'w') as f:
                    json.dump(self[i].to_htdict(), f)
                    if verbose:
                        print(f'Saved to "{numbered_file}"')



    def to_csv(self,file,header=None,**kwargs):
        """Dump run to CSV file in HEPDATA format

        Parameters
        ----------
        file : str
            Output file.

        **kwargs
            - header: specify header for csv file
            - all_values: print not only central value, upper and lower band values,
              but across all setups
            - logx: if true, bin centers are geometric mean, otherwise simple average

        Returns
        -------
        None
        """
        df = pd.DataFrame()
        if self.dim() == 1:
            def centers(edges,logx):
                if (logx):
                    return [(l*r)**0.5 for l,r in zip(edges[:-1],edges[1:])]
                else:
                    return [(l+r)*.5 for l,r in zip(edges[:-1],edges[1:])]
            df['BIN'] = centers(self.edges[0], kwargs.get('logx'))
            df['BIN LOW'] = self.edges[0][:-1]
            df['BIN HIGH'] = self.edges[0][1:]
            df['VALUE [PB]'] = self.v()
            df['ERROR+'] = self.e()
            df['ERROR-'] = -self.e()
            if (kwargs.get('all_values', False)):
                for i in range(self.nsetups()):
                    df[f'VAL{i}'] = self.values[:,i]
            else:
                df['SYS+'] = self.upper() - self.v()
                df['SYS-'] = self.lower() - self.v()
        else:
            raise Exception("Multi dimensional data dump to CSV not supported yet")

        with open(file, 'w') as f:
            if header: f.write(f'{header}\n')
            if 'obs' in self.info: f.write(f'# Observable: {self.info["obs"]}\n')
            if 'process' in self.info: f.write(f'# Process: {self.info["process"]}\n')
            if self.info.get('variation',[]):
                f.write('# Central setup: {}\n'\
                        .format(self.info.get("variation",'')[0]))

            df.to_csv(f, index=False, float_format="%.6e",
                      header=kwargs.get('header',True))
            print(f'Run saved to: {file}')


    @staticmethod
    def convert_to_edges(binsList):
        """Get edges for each dimension given a list of bins

        Parameters
        ----------
        binsList : list
            Bins in the format of 3D-list: [... [ [l1,r1], [l2,r2], ...], ...]

        Returns
        -------
        list
            2-dimensional list.
        """
        if len(binsList[0]) == 1:
            return [[ binsList[0][0][0] ] + [ bins[0][1] for bins in binsList ]]
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
        """Get full list of bins given edges for each dimension

        Parameters
        ----------
        edgesList : list
            Bin edges in the format of 2D-list: [ ... , [x0, x1 ... ], ... ]

        Returns
        -------
        list
            3-dimensional list.
        """
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
    def full(dims, nsetups=1, fill_value=0):
        """Get run with filled const values

        Parameters
        ----------
        nsetups : int, default: 1
            Number of scale setups.

        fill_value : float, default: 0
            Constant value to fill into histograms.

        Returns
        -------
        Run
        """
        run = Run()
        run.edges = [list(range(d+1)) for d in dims if d > 0]
        run.values = np.full((len(run.bins),nsetups),float(fill_value))
        run.errors = np.full((len(run.bins),nsetups),0.)
        return run


    @staticmethod
    def seq(dims, nsetups=1):
        """Get multidimensional run for testing

        Values are taken from the natural sequence.

        Parameters
        ----------
        nsetups : int, default: 1
            Number of scale setups.

        Returns
        -------
        Run
        """
        run = Run()
        run.edges = [list(range(d+1)) for d in dims if d > 0]
        run.values = np.arange(0,len(run.bins),1./nsetups)\
                    .reshape(len(run.bins),nsetups)
        run.errors = run.values / 10
        return run


    @staticmethod
    def random(dims, nsetups=1, seed=None):
        """Get multidimensional run for testing

        Values are generated randomly.

        Parameters
        ----------
        optional nsetups : int, default: 1
            Number of scale setups.

        optional seed : int, default: None
            Set seed in numpy.random.

        Returns
        -------
        Run
        """
        if seed:
            np.random.seed(seed)
        run = Run()
        run.edges = [list(range(d+1)) for d in dims if d > 0]
        run.values = np.random.rand(len(run.bins),nsetups)
        run.errors = np.random.rand(len(run.bins),nsetups) / 10
        return run


    # TODO: nice printout
    def __repr__(self):
        """Full printout for the run.

        Returns
        -------
        str
            String containing all attributes.
        """
        desc = ""
        for m in self._get_attributes():
            desc += f" '{m}': {getattr(self,m)}\n"
        return desc
