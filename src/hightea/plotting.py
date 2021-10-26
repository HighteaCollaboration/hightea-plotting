import pathlib
import numpy as np
from functools import wraps
import matplotlib.pyplot as plt
from .run import Run

###########################
#  Matplotlib parameters  #
###########################
plt.rcParams['axes.xmargin'] = 0
colorscheme = ['black', '#1f77f4', '#2ca02c',
               '#d62728', '#e377c2', '#8c564b',
               '#1f77b4', '#9467bd', '#17becf']

def _select_keys(d, *args):
    return {k:d[k] for k in d.keys() if k in args}

def _convert_args_to_Runs(plotfunc):
    """Convert JSON files and dicts to Run class"""
    @wraps(plotfunc)
    def inner(*args, **kwargs):
        runs = []
        for a in args:
            if not(isinstance(a, Run)):
                run = Run()
                run.load(a)
                runs.append(run)
            else:
                runs.append(a)
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
    _showRatio = not(_ratio == None)

    plt.suptitle(kwargs.get('title', runs[0].meta.get('obs')))

    ax = fig.add_subplot(3, 1, (1, 2)) if (_showRatio) else plt.gca()
    plot_histograms_1d(ax, *runs, **kwargs)

    if (_showRatio):
        ax = fig.add_subplot(3, 1, 3)
        ratio_runs = []
        for i,r in enumerate(runs):
            ratio_runs.append(runs[i] / runs[_ratio][0])
        plot_histograms_1d(ax, *ratio_runs, **kwargs, legend=False)

    plt.tight_layout()

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


def plot_histograms_1d(ax, *runs, **kwargs):
    """Procedure to draw 1d histograms given ax handle"""

    _showScaleBand = kwargs.get('showScaleBand', True)
    _showErrors = kwargs.get('showErrors', True)
    _showGrid = kwargs.get('grid', True)
    _colorscheme = kwargs.get('colorscheme',colorscheme)
    _showLegend = kwargs.get('legend', True)

    for i,run in enumerate(runs):
        if (run.dim() != 1):
            raise Exception('2D histograms not supported yet')

        def m(a):
            return list(a)+[a[-1]]

        _edges = run.edges[0]

        ax.step(_edges,
                m(run.v()),
                where='post',
                color=_colorscheme[i],
                label=run.meta.get('name',f'run {i}'),
                **_select_keys(kwargs,'linewidth','alpha'))

        if (_showScaleBand):
            ax.fill_between(_edges,
                            m(run.lower()),
                            m(run.upper()),
                            step='post',
                            linewidth=0.0,
                            color=_colorscheme[i],
                            alpha=0.2)

        if (_showErrors):
            errXshift = .03*(i-(len(runs)-1)/2)
            errXs = (.5 + errXshift)*_edges[1:] +\
                    (.5 - errXshift)*_edges[:-1]
            ax.errorbar(errXs,
                        run.v(),
                        yerr=run.e(),
                        color=_colorscheme[i],
                        linestyle='')

    if (_showGrid):
        ax.grid(lw=0.2, c='gray')

    if (_showLegend):
        ax.legend()


