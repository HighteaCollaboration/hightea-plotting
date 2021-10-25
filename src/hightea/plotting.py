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
    _showGrid = kwargs.get('grid', True)
    _show = kwargs.get('show', True)
    _output = kwargs.get('output', None)
    _ratio = kwargs.get('ratio', None)

    plt.suptitle(kwargs.get('title', runs[0].meta.get('obs')))

    ax = plt.gca()

    # ref = runs[0].mincopy()
    # ratio_runs = []
    # for i,r in enumerate(runs):
    #     ratio_runs.append(runs[i] / ref)
    # plot_histograms_1d(ax, *ratio_runs, **kwargs)
    plot_histograms_1d(ax, *runs, **kwargs)

    if (_ratio):
        axes[i]

    if (_showLegend):
        ax.legend()
        pass

    if (_showGrid):
        ax.grid(lw=0.2, c='gray')

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
    _colorscheme = kwargs.get('colorscheme',colorscheme)

    for i,run in enumerate(runs):
        if (run.dim() != 1):
            raise Exception('2D histograms not supported yet')

        def m(a):
            return list(a)+[a[-1]]

        ax.step(run.edges[0],
                m(run.values[:,0]),
                where='post',
                color=_colorscheme[i],
                label=run.meta.get('name',f'run {i}'),
                **_select_keys(kwargs,'linewidth','alpha'))

        if (_showScaleBand):
            ax.fill_between(run.edges[0],
                            m(np.amin(run.values, axis=1)),
                            m(np.amax(run.values, axis=1)),
                            step='post',
                            linewidth=0.0,
                            color=_colorscheme[i],
                            alpha=0.2)

        if (_showErrors):
            errXshift = .03*(i-(len(runs)-1)/2)
            errXs = (.5 + errXshift)*run.edges[0][1:] +\
                    (.5 - errXshift)*run.edges[0][:-1]
            ax.errorbar(errXs,
                        run.values[:,0],
                        yerr=run.errors[:,0],
                        color=_colorscheme[i],
                        linestyle='')

