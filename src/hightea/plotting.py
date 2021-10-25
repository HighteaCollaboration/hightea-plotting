import pathlib
import numpy as np
from functools import wraps
import matplotlib.pyplot as plt
from .run import Run

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
    fig, axes = plt.subplots(**_select_keys(kwargs,'figsize'))

    _showScaleBand = kwargs.get('showScaleBand', True)
    _showErrors = kwargs.get('showErrors', True)
    _showLegend = kwargs.get('legend', True)
    _show = kwargs.get('show', True)

    ax = axes[0] if (isinstance(axes,list)) else axes

    for i,run in enumerate(runs):
        if (run.dim != 1):
            raise Exception('2D histograms not supported yet')

        def m(a):
            return list(a)+[a[-1]]

        ax.step(run.edges[0],
                m(run.values[:,0]),
                where='post',
                **_select_keys(kwargs,'linewidth','alpha','label'))

        if (_showScaleBand):
            ax.fill_between(run.edges[0],
                            m(np.amin(run.values, axis=1)),
                            m(np.amax(run.values, axis=1)),
                            step='post',
                            linewidth=0.0,
                            alpha=0.2)

        if (_showErrors):
            errXshift = .03*(i-(len(runs)-1)/2)
            errXs = (.5 + errXshift)*run.edges[0][1:] + (.5 - errXshift)*run.edges[0][:-1]
            ax.errorbar(errXs,
                        run.values[:,0],
                        yerr=run.errors[:,0],
                        linestyle='')

        if (_showLegend):
            ax.legend()
            pass

    # TODO: legend

    if (_show):
        plt.show()

    plt.clf()


