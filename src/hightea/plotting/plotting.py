import numpy as np
from functools import wraps
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from .run import Run

###########################
#  Matplotlib parameters  #
###########################
plt.rcParams['axes.xmargin'] = 0
colorscheme = ['tab:blue', 'tab:green', 'tab:red',
               'tab:pink', 'tab:purple', 'tab:cyan',
               'tab:orange', 'tab:olive', 'tab:brown']

def _select_keys(d, *args):
    return {k:d[k] for k in d.keys() if k in args}

def _get_info(runs, *args):
    """Scroll through runs to get relevant info to show on plots"""
    res = {}
    for a in args:
        i = 0
        while not(a in res) and i < len(runs):
            if a in runs[i].info:
                res[a] = runs[i].info.get(a)
            i += 1
    return res

def _convert_args_to_Runs(plotfunc):
    """Convert JSON files and dicts to Run class"""
    @wraps(plotfunc)
    def inner(*args, **kwargs):
        runs = []
        def _convert_to_runs(runs, obj):
            if isinstance(obj, list) or isinstance(obj, tuple):
                for o in obj:
                    _convert_to_runs(runs, o)
            else:
                if isinstance(obj,str) or isinstance(obj,dict):
                    runs.append(Run(obj))
                else:
                    runs.append(obj)
        for a in args:
            _convert_to_runs(runs, a)
        return plotfunc(*runs, **kwargs)
    return inner


@_convert_args_to_Runs
def plot(*runs, **kwargs):
    """General plotting routine

    Parameters
    ----------
    *runs
        Objects that are instances of Run class, or can be cast into one
        (filepaths as strings, dictionaries), passed as positional arguments.

    **kwargs
        Optional keyword arguments

        - show : bool, default: True
            Show plot after being done.

        - ratio : int
            Add ratio plot w.r.t run under provided id.

        - figsize : tuple
            Specify size of figure in the same way as matplotlib.

        - logscale : bool
            Plot on logscale over axes of interest.
            Possible values: 'x', 'y', 'xy'.

        - lim : dict
            Specify limits on axes, e.g {'y2': [0.8,1.2]}.

        - latex : bool, default: False
            Use latex fonts on plots (warning: slow).

        - title : str
            Specify plot title.

        - show_setup : bool
            Show information on central setup.

        - output : str
            Save to png/pdf file if specified.

        - figure : matplotlib.pyplot.Figure
            If passed, use as canvas, otherwise create new.

    Examples
    --------
    >>> # Import library
    >>> import hightea.plotting as ht

    >>> # Plot data stored in files
    >>> ht.plot('HEPDATA-experiment.csv', 'request.json')

    >>> # Construct and plot random runs
    >>> run = ht.Run.random([10], nsetups=3)
    >>> run2 = run.minicopy()
    >>> run2 += 1
    >>> ht.plot(run, run2, ratio=0, figsize=(16,9), title='Random runs')

    >>> # Example of plotting on top of the data
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> runs = ...
    >>> ht.plot(*runs, figure=fig, show=False)
    >>> ax = fig.get_axes()[0]
    >>> X = np.array([(l+r)/2 for l,r in zip(run.edges[0][:-1], run.edges[0][1:])])
    >>> Y = (X**2+2*80**2)**-2
    >>> ax.scatter(X, Y, label='fit')
    >>> ax.legend()
    >>> plt.show()

    Note
    ----
    See supporting procedures to see how other keyword arguments are used.

    Returns
    -------
    matplotlib.pyplot.Figure
        Figure which used in plotting to allow further manipulations.
    """
    _fig = kwargs.get('figure')
    _show = kwargs.get('show', True)
    _output = kwargs.get('output', None)
    _ratio = kwargs.get('ratio', None)
    _logscale = kwargs.get('logscale', None)
    _lim = kwargs.get('lim', {})
    _latex = kwargs.get('latex', None)
    _showRatio = not(_ratio == None)
    _showSetup = kwargs.get('show_setup', None)
    _info = _get_info(runs, *'obs binning process variation'.split())

    if not _latex is None:
        if _latex:
            plt.rc('font', family='CMU Serif', serif=['Roman'], size=14)
            plt.rc('text', usetex=True)
            plt.rc('text.latex',
                    preamble=r'\usepackage{amsmath}\usepackage{amssymb}')
        else:
            plt.style.use('default')

    if _fig is None:
        if _show:
            _fig = plt.figure(**_select_keys(kwargs,'figsize'))
        else:
            _fig = Figure(**_select_keys(kwargs,'figsize'))

    obs = _info.get('obs','')
    binning = _info.get('binning',[])

    if (_logscale == None):
        for k in 'transverse energy mass'.split():
            if (k in obs):
                _logscale = True

    _fig.suptitle(kwargs.get('title', obs))
    axes = _fig.get_axes()
    if axes:
        ax1 = axes[0]
    else:
        ax1 = _fig.add_subplot(3, 1, (1, 2)) if (_showRatio) else _fig.gca()
    plot_unrolled(ax1, *runs, **kwargs)

    if (_logscale):
        if (isinstance(_logscale,str)):
            ax1.set_xscale('log') if 'x' in _logscale else ...
            ax1.set_yscale('log') if 'y' in _logscale else ...
        else:
            ax1.set_yscale('log')

    if (_showRatio):
        if len(axes) > 1:
            ax2 = axes[1]
        else:
            ax2 = _fig.add_subplot(3, 1, 3, sharex = ax1)
        ax1.get_xaxis().set_visible(False)
        ratio_runs = []
        for i,r in enumerate(runs):
            ratio_runs.append(runs[i] / runs[_ratio].v())
        plot_unrolled(ax2, *ratio_runs, **kwargs, legend=False)
        ylim = ax2.get_ylim()
        ax2.set_ylim(max(ylim[0], -10), min(ylim[1], 10))
        ax2.set_ylabel('Ratio')
        if (_lim):
            if ('x2' in _lim): ax2.set_xlim(_lim.get('x2'))
            if ('y2' in _lim): ax2.set_ylim(_lim.get('y2'))
        _fig.set_tight_layout(True)

    if (binning):
        obslabel = binning[0].get('variable')
        ax1.set_xlabel(obslabel)
        # TODO: put labels on top of the picture for higher-dim plots
        sigmaletter = 'σ' if not _latex else '$\\sigma$'
        units = '[pb/X]' if runs[0].is_differential() else '[pb]'
        ax1.set_ylabel(f'd{sigmaletter} / d({obslabel}) {units}')

    if (_lim):
        if ('x1' in _lim): ax1.set_xlim(_lim.get('x1'))
        if ('y1' in _lim): ax1.set_ylim(_lim.get('y1'))

    if (_showSetup) or (len(runs) == 1 and (_showSetup == None)):
        headerinfo = []
        headerinfo.append('Process: '+_info.get("process")) if "process" in _info else ...
        headerinfo.append('Central setup: '+_info.get("variation",'')[0]) \
                          if len(_info.get('variation',[])) else ...
        if (headerinfo):
            ax1.text(.02,.98, (5*' ').join(headerinfo),
                      bbox = dict(facecolor='white',alpha=.6,linewidth=.5),
                      verticalalignment = 'top',
                      transform=ax1.transAxes)

    if (_output):
        ext = _output.split('.')[-1]
        if (ext == 'pdf'):
            pp = PdfPages(_output); pp.savefig(); pp.close()
        elif (ext == 'png'):
            _fig.savefig(_output)
        else:
            raise Exception("Unexpected extension")
        print(f'Figure saved to: {_output}')

    if (_show):
        plt.show()

    return _fig


def plot_unrolled(ax, *runs, **kwargs):
    """Procedure to plot provided runs as 1D unrolled histograms

    Parameters
    ----------
    *runs
        Runs passed as positional arguments.

    **kwargs
        - grid : bool, default: True
            If True, show grid on plot.

        - colorscheme : list
            List of colors to use for provided runs.

        - legend : bool
            If True, show legend.

        - finetune : dict
            Passes parameters (values) directly to plotting functions (keys).
            Passes on to underlying local plotting routines.
            Affects: `Axes.grid`.

    Examples
    --------
    >>> # Example of a more involved plot using underlying routines
    >>> hightea.plotting as ht
    >>> matplotlib.pyplot as plt
    >>> run = ht.Run.random([10], nsetups=3)
    >>> runs = dict(lo=run, data=run+0.5, nnlo=(run+0.6).update_info('NNLO absolute'))
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(2,1,1)
    >>> ht.plotting.plot_unrolled(ax, runs['nnlo'])
    >>> ax = fig.add_subplot(2,1,2)
    >>> ht.plotting.plot_unrolled(ax, *[
    >>>         (runs['lo']/runs['data'].v())[0].update_info('LO (central scale)'),
    >>>         (runs['data']/runs['data'].v()).update_info(name='NLO', experiment=True),
    >>>         (runs['nnlo']/runs['nlo'].v()).update_info('NNLO'),
    >>>     ],
    >>>    finetune=dict(grid={"alpha":.3}),
    >>> )
    >>> ax.set_ylabel('Ratio to data')
    >>> plt.show()

    Note
    ----
    See supporting procedures to see how other keyword arguments are used.

    Returns
    -------
    None
    """
    if not len(runs):
        print("No runs provided, exitting")
        return

    _showGrid = kwargs.get('grid', True)
    _colorscheme = kwargs.get('colorscheme',colorscheme)
    _showLegend = kwargs.get('legend', True)
    _finetune = kwargs.get('finetune', {})

    # plot each run separately
    for i,run in enumerate(runs):

        color = _colorscheme[i % len(_colorscheme)]

        # separate treatment for experimental data and theoretical distributions
        if not(run.info.get('experiment',False)):
            _plot_theory(ax,run.remove_OUF(),**kwargs,
                            color=color,
                            label=f'run {i}' if run.name==None else run.name,
                            errshift=.03*(i-(len(runs)-1)/2))

        else:
            _plot_experiment(ax,run.remove_OUF(),**kwargs,
                            color=color,
                            label=f'data {i}' if run.name==None else run.name)

        # put OUF bins on plot if they exist
        if run.has_OUF() and run.dim() == 1:

            def get_oufrun(run, i):
                if (i < 0):
                    dx = run.edges[0][i-1]-run.edges[0][i-2]
                    lx = run.edges[0][i-1] + dx*0.2
                    rx = run.edges[0][i-1] + dx*1.2
                else:
                    dx = run.edges[0][i+2]-run.edges[0][i+1]
                    lx = run.edges[0][i+1] - dx*1.2
                    rx = run.edges[0][i+1] - dx*0.2
                oufrun = Run(bins=[[[lx,rx]]])
                oufrun.values = np.array([list(run.values[i])*2])
                oufrun.errors = np.array([list(run.errors[i])*2])
                return oufrun

            OUFkwargs = dict(**kwargs,label=None,color=color,
                             errshift=.03*(i-(len(runs)-1)/2))

            if abs(run.edges[0][-1]) == float('inf'):
                _plot_theory(ax,get_oufrun(run,-1),**OUFkwargs,marker='4')
            if abs(run.edges[0][0]) == float('inf'):
                _plot_theory(ax,get_oufrun(run,0),**OUFkwargs,marker='3')


    # show dimension edges for multidimensional distributions
    if (runs[0].dim() > 1):
        run = runs[0].remove_OUF()
        for j in range(1,run.dimensions()[0]):
            ax.axvline(run.edges[1][0] +
                        j*(run.edges[1][-1] - run.edges[1][0]),
                        ls=':', color='gray')

    if (_showGrid):
        ax.grid(**{
                    'lw': 0.2,
                    'alpha': 1,
                    'c': 'gray',
                    **_finetune.get('grid',{})
                })

    if (_showLegend):
        ax.legend(loc=kwargs.get('legend_loc','upper right'))


def _get_unrolled(edges):
    """Unroll edges of 2D run into bins of 1D run

    Parameters
    ----------
    edges : list

    Returns
    -------
    list
    """
    if len(edges) == 1:
        return np.array(edges[0])
    elif len(edges) == 2:
        unrolled = edges[1].copy()
        dims = [len(x)-1 for x in edges]
        for i in range(1,dims[0]):
            unrolled.extend(list(np.array(edges[1][1:])
                            + i*(edges[1][-1] - edges[1][0])))
        unrolled = np.array(unrolled)
        return unrolled


def _plot_theory(ax,run,**kwargs):
    """Procedure to plot theoretical observable given axis handle

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        Axes instance to plot on.

    run : Run
        Run instance to use for plotting.

    **kwargs
        - color : str

        - errshift : float, default: 0

        - showScaleBand : bool, default: True

        - showErrors : bool, default: True

        - linewidth : float, default: 2.5

        - linestyle : str, default: '-'

        - alpha : float, default: 0.3

        - marker : str, default: ''

        - ms : float, default: 20

        - finetune : dict
            Passes parameters (values) directly to matplotlib functions (keys).
            Affects: `Axes.fill_between`.

    Returns
    -------
    None
    """
    def m(a):
        return list(a)+[a[-1]]
    _edges = _get_unrolled(run.edges)
    _color = kwargs.get('color')
    _errshift = kwargs.get('errshift',0)
    _showScaleBand = kwargs.get('showScaleBand', True)
    _showErrors = kwargs.get('showErrors', True)
    _linewidth = kwargs.get('linewidth', 2.5)
    _linestyle = kwargs.get('linestyle', '-')
    _alpha = kwargs.get('alpha', .3)
    _marker = kwargs.get('marker', '')
    _ms = kwargs.get('ms', 20)
    _finetune = kwargs.get('finetune', {})

    ax.step(_edges,
            m(run.v()),
            where='post',
            color=_color,
            marker=_marker,
            ms=_ms,
            label=kwargs.get('label'),
            linestyle=_linestyle,
            linewidth=_linewidth)

    if (_showScaleBand):
        """ Several ways to show scale band """
        if _showScaleBand in (True, 'fill'):
            ax.fill_between(_edges,
                            m(run.lower()),
                            m(run.upper()),
                            **{
                                'step': 'post',
                                'linewidth': 0.0,
                                'color': _color,
                                'alpha': _alpha,
                                **_finetune.get('fill_between', {}),
                            })
        else:
            for y in m(run.lower()), m(run.upper()):
                ax.step(_edges, y,
                        where='post',
                        linestyle=_showScaleBand,
                        linewidth=.7*_linewidth,
                        color=_color,
                        alpha=_alpha)

    if (_showErrors):
        errXs = (.5 + _errshift)*_edges[1:] +\
                (.5 - _errshift)*_edges[:-1]
        ax.errorbar(errXs,
                    run.v(),
                    yerr=run.e(),
                    color=_color,
                    linestyle='')


def _plot_experiment(ax,run,**kwargs):
    """Support function to plot experimental observable

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        Axes instance to plot on.

    run : Run
        Run instance to use for plotting.

    **kwargs
        - color : str

        - errshift : float, default: 0

        - marker : str, default: 'o'

        - ms : float, default: 3

        - capsize : float, default: 5

        - label : str

        - linewidth : float, default: 2.5

    Returns
    -------
    None
    """
    _edges = np.array(_get_unrolled(run.edges))
    _xs = np.array([.5*(l+r) for l,r in zip(_edges[:-1],_edges[1:])])
    _color = kwargs.get('color')
    _errshift = kwargs.get('errshift',0)
    _marker = kwargs.get('marker', 'o')
    _ms = kwargs.get('ms', 3)
    _capsize = kwargs.get('capsize', 5.)
    _label = kwargs.get('label')
    _linewidth = kwargs.get('linewidth', 2.5)

    ax.errorbar(x=_xs,
                y=run.v(),
                yerr=[run.v()-run.lower(), run.upper()-run.v()],
                label=_label,
                marker=_marker,
                ms=_ms,
                color=_color,
                capsize=_capsize,
                linestyle='None',
                linewidth=_linewidth)

