from math import isclose
from src.hightea.run import Run
import numpy as np
from copy import deepcopy

def test_RunClass_create():
    run = Run()

# test various loading methods
def test_loading_dict():
    run = Run()
    run.load({'mean': [[[[0,1]],[1.0]], [[[1,2]],[2.0]], [[[2,3]],[3.0]]],
              'std':  [[[[0,1]],[0.1]], [[[1,2]],[0.2]], [[[2,3]],[0.3]]]})
    assert((run.edges[0] == np.array([0,1,2,3])).all())

def test_loading_json():
    run = Run()
    run.load('tests/input/simple1d.json')
    assert(run.bins[0] == [[0,1]])
    assert(run.is_differential())

# test conversion from bins to edges
def test_convert_to_edges_1d():
    bins = [[[0,1]],[[1,2]],[[2,3]]]
    result = Run.convert_to_edges(bins)
    expected = [np.array([0,1,2,3])]
    for b,e in zip(result, expected):
        assert((b == e).all())

def test_convert_to_edges_2d():
    bins = [[[0,1],[0,1]],[[0,1],[1,2]],[[1,2],[0,1]],[[1,2],[1,2]]]
    result = Run.convert_to_edges(bins)
    expected = [np.array([0,1,2]),np.array([0,1,2])]
    for b,e in zip(result, expected):
        assert((b == e).all())

    bins = [[[0,1],[0,1]],[[0,1],[1,2]]]
    result = Run.convert_to_edges(bins)
    expected = [np.array([0,1]),np.array([0,1,2])]
    for b,e in zip(result, expected):
        assert((b == e).all())

# test conversion from edges to bins
def test_convert_to_bins_1d():
    edges = [np.array([0,1,2,3])]
    result = Run.convert_to_bins(edges)
    expected = [[[0,1]],[[1,2]],[[2,3]]]
    assert(np.array_equal(result, expected))

def test_convert_to_bins_2d():
    edges = [np.array([0,1,3]),np.array([1,2])]
    result = Run.convert_to_bins(edges)
    expected = [[[0,1],[1,2]], [[1,3],[1,2]]]
    assert(np.array_equal(result, expected))

# test obtaining subarea of the run binnings
def test_slicing_1d():
    run = Run()
    run.load({'mean': [[[[0,1]],[1.0]], [[[1,2]],[2.0]], [[[2,3]],[3.0]]],
              'std':  [[[[0,1]],[0.1]], [[[1,2]],[0.2]], [[[2,3]],[0.3]]]})
    newrun = run[1:]
    assert((run.edges[0] == np.array([0,1,2,3])).all())
    assert((newrun.edges[0] == np.array([1,2,3])).all())
    assert(len(newrun.bins) == len(newrun.values))

def test_slicing_2d():
    run = Run(bins = [[[0,1],[0,1]],[[0,1],[1,2]],[[1,2],[0,1]],[[1,2],[1,2]]])
    run.values = np.ones((len(run.bins),1))
    newrun = run[1]
    assert((newrun.edges[0] == np.array([1,2])).all())
    assert(len(newrun.bins) == len(newrun.values))
    newrun.flatten()
    assert(len(newrun.edges) == 1)

def test_random():
    run = Run.random((2,3,1),3)
    expected = [np.array([0,1,2]), np.array([0,1,2,3]), np.array([0,1])]
    for r,e in zip(run.edges, expected):
        assert((r == e).all())
    assert(run.values.shape == (len(run.bins),3))
    assert(run.errors.shape == (len(run.bins),3))

def test_rescaling_1d():
    run = Run.random((2,3),3)
    divrun = run[:]
    divrun.rescale(run.values,correlated=True)
    assert(np.isclose(divrun.values.flatten(),np.ones(2*3*3)).all())




