import pathlib
import numpy as np
from functools import wraps
import matplotlib.pyplot as plt
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
        plotfunc(*runs, **kwargs)
    return inner


@_convert_args_to_Runs
def plot(*runs, **kwargs):
    """General plotting routine for 1d histograms"""
    fig = plt.figure(**_select_keys(kwargs,'figsize'))

    _showLegend = kwargs.get('legend', True)
    _show = kwargs.get('show', True)
    _output = kwargs.get('output', None)
    _ratio = kwargs.get('ratio', None)
    _logscale = kwargs.get('logscale', None)
    _showRatio = not(_ratio == None)
    _info = _get_info(runs, *'obs binning'.split())

    obs = _info.get('obs','')
    binning = _info.get('binning',[])

    if (_logscale == None):
        for k in 'transverse energy mass'.split():
            if (k in obs):
                _logscale = True

    plt.suptitle(kwargs.get('title', obs))

    ax1 = fig.add_subplot(3, 1, (1, 2)) if (_showRatio) else plt.gca()
    plot_unrolled(ax1, *runs, **kwargs)

    if (_logscale):
        if (isinstance(_logscale,str)):
            ax1.set_xscale('log') if 'x' in _logscale else ...
            ax1.set_yscale('log') if 'y' in _logscale else ...
        else:
            ax1.set_yscale('log')

    if (_showRatio):
        ax2 = fig.add_subplot(3, 1, 3, sharex = ax1)
        ax1.get_xaxis().set_visible(False)
        ratio_runs = []
        for i,r in enumerate(runs):
            ratio_runs.append(runs[i] / runs[_ratio][0])
        plot_unrolled(ax2, *ratio_runs, **kwargs, legend=False)
        ylim = ax2.get_ylim()
        ax2.set_ylim(max(ylim[0], -10), min(ylim[1], 10))
        ax2.set_ylabel('Ratio')
        plt.tight_layout()

    if (binning):
        obslabel = binning[0].get('variable')
        plt.xlabel(obslabel)
        # TODO: put labels on top of the picture for higher-dim plots
        ax1.set_ylabel(f'dσ / d({obslabel}) [pb/X]')

    # TODO: add info on top about process, scales, PDFs

    if (_output):
        ext = _output.split('.')[-1]
        if (ext == 'pdf'):
            pp = PdfPages(_output); pp.savefig(); pp.close()
        elif (ext == 'png'):
            plt.savefig(_output)
        else:
            raise Exception("Unexpected extension")
        print(f'Figure saved to: {_output}')

    if (_show):
        plt.show()

    plt.clf()


def plot_unrolled(ax, *runs, **kwargs):
    """Procedure to draw 1d runs"""

    _showGrid = kwargs.get('grid', True)
    _colorscheme = kwargs.get('colorscheme',colorscheme)
    _showLegend = kwargs.get('legend', True)

    for i,run in enumerate(runs):
        # if (run.dim() != 1):
        #     raise Exception('2D histograms not supported yet')

        if not(run.info.get('experiment',False)):
            _plot_theory(ax,run,**_select_keys(kwargs,'linewidth','alpha'),
                            color=_colorscheme[i],
                            label=f'run {i}' if run.name==None else run.name,
                            errshift=.03*(i-(len(runs)-1)/2))
        else:
            _plot_experiment(ax,run,**_select_keys(kwargs,'linewidth','alpha'),
                            color=_colorscheme[i],
                            label=f'data {i}' if run.name==None else run.name)

        if (i == 0 and run.dim() > 1):
            for j in range(1,run.dimensions()[0]):
                ax.axvline(run.edges[1][0] + j*(run.edges[1][-1] - run.edges[1][0]),
                            ls=':', color='gray')

    if (_showGrid):
        ax.grid(lw=0.2, c='gray')

    if (_showLegend):
        ax.legend()


def _get_unrolled(edges):
    """Return just bins or unrolled bins for 2d plot"""
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
    """Support function to plot theoretical observable given ax handle"""
    def m(a):
        return list(a)+[a[-1]]
    _edges = _get_unrolled(run.edges)
    _color = kwargs.get('color')
    _errshift = kwargs.get('errshift',0)
    _showScaleBand = kwargs.get('showScaleBand', True)
    _showErrors = kwargs.get('showErrors', True)
    _linewidth = kwargs.get('linewidth', 2.5)
    _alpha = kwargs.get('alpha', .3)

    ax.step(_edges,
            m(run.v()),
            where='post',
            color=_color,
            label=kwargs.get('label'),
            linewidth=_linewidth)

    if (_showScaleBand):
        ax.fill_between(_edges,
                        m(run.lower()),
                        m(run.upper()),
                        step='post',
                        linewidth=0.0,
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
    """Support function to plot experimental observable"""
    _edges = np.array(_get_unrolled(run.edges))
    _xs = np.array([.5*(l+r) for l,r in zip(_edges[:-1],_edges[1:])])
    _color = kwargs.get('color')
    _errshift = kwargs.get('errorshift',0)
    _showScaleBand = kwargs.get('showScaleBand', True)
    _showErrors = kwargs.get('showErrors', True)
    _marker = kwargs.get('marker', 'o')
    _ms = kwargs.get('ms', 3)
    _capsize = kwargs.get('capsize', 5.)
    _label = kwargs.get('label')
    _linewidth = kwargs.get('linewidth', 2.5)
    _alpha = kwargs.get('alpha', .3)

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

