import pytest
from math import isclose
from src.hightea.plotting.run import Run
import numpy as np
from copy import deepcopy
from pathlib import Path
import os
import shutil

def test_RunClass_create():
    run = Run()

#################
#  I/O methods  #
#################

def test_loading_dict():
    run = Run()
    run.load({'histogram': [
        {'edges':[[0, 1]],'mean':[0], 'error':[0]},
        {'edges':[[1, 2]],'mean':[1], 'error':[0]},
        {'edges':[[2, 3]],'mean':[2], 'error':[0]},
        ]})
    assert((run.edges[0] == np.array([0,1,2,3])).all())
    run.load({'histogram': [
        {'edges':[{'min_value':0,'max_value':1}],'mean':[0], 'error':[0]},
        {'edges':[{'min_value':1,'max_value':2}],'mean':[1], 'error':[0]},
        {'edges':[{'min_value':2,'max_value':3}],'mean':[2], 'error':[0]},
        ]})
    assert((run.edges[0] == np.array([0,1,2,3])).all())
    assert((run.v() == np.array([0,1,2])).all())
    # with pytest.warns(UserWarning):
    run.load({'histogram': [
        {'edges':[{'min_value':0,'max_value':1}],'mean':0},
        {'edges':[{'min_value':1,'max_value':2}],'mean':1},
        {'edges':[{'min_value':2,'max_value':3}],'mean':2},
        ]})
    assert((run.v() == np.array([0,1,2])).all())

def test_loading_json():
    run = Run()
    run.load('tests/input/simple1d.json')
    assert run.bins[0] == [[0,1]]
    assert run.is_differential()

def test_dumping_json():
    run = Run.seq((4,5),nsetups=5)
    wdir = 'tests/tmp'
    if not os.path.isdir(wdir):
        os.makedirs(wdir)
    run.to_json(Path(wdir,'run.json'),verbose=False)
    new = Run(Path(wdir,'run-0.json'))
    assert new.nsetups() == 1
    run.to_json(Path(wdir,'run.json'),verbose=False,combined=True)
    new = Run(Path(wdir,'run.json'))
    assert new.nsetups() == run.nsetups()
    shutil.rmtree(wdir)

def test_loading_hepdata():
    run = Run()
    run.load('tests/input/HEPData-ins1610623-v1-Table_7.csv')
    assert run.dim() == 1
    assert isclose(run.edges[0][-1], 2.4)

def test_makehistogram_json():
    run = Run('tests/input/simple2d.json')
    newrun = deepcopy(run)
    run.make_histogramlike()
    run.make_differential()
    assert(run == newrun)

##############################
#  bin to edges conversions  #
##############################

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

#####################
#  slicing methods  #
#####################

def test_slicing_scale():
    run = Run.random((2,3),4)
    assert((run[1:].nsetups()) == 3)
    assert((run[0].nsetups()) == 1)

def test_random():
    shape = [2,3,1]
    run = Run.random(shape,3)
    assert(run.dimensions() == shape)
    expected = [np.array([0,1,2]), np.array([0,1,2,3]), np.array([0,1])]
    for r,e in zip(run.edges, expected):
        assert((r == e).all())
    assert(run.values.shape == (len(run.bins),3))
    assert(run.errors.shape == (len(run.bins),3))

def test_zoom():
    # TODO: update test for slices at other dimensions
    run = Run.seq((2,3),2)
    run.edges = [np.array([-5,-3,-1]), np.array(list(range(4)))]
    run2 = run.zoom(-2)
    assert((run2.values[:,0] == np.array([3,4,5])).all())
    run2 = run.zoom(line=1)
    assert((run2.values[:,0] == np.array([3,4,5])).all())

#########################
#  ratio and rescaling  #
#########################

def test_div():
    run = Run.random((2,3),3)
    divrun = run[:]
    newrun = divrun / run
    assert(np.isclose(newrun.values.flatten(),np.ones(2*3*3)).all())

######################
#  Passing metadata  #
######################

def test_name():
    run = Run('tests/input/simple2d.json', name='testname')
    assert(run.name == 'testname')

##########################
#  Arithmetic operators  #
##########################

def test_eq():
    run = Run.random((2,3),3)
    new = run.minicopy()
    # TODO revise this test
    assert(new == run)

def test_add():
    run = Run.random((2,3),nsetups=3)
    zeros = Run.full((2,3),nsetups=3)
    new = zeros + run
    assert(new == run)
    assert(2 - 2 + zeros + 1 - 1 == zeros)

def test_muldiv():
    old = Run.random((2,3),nsetups=3)
    new = deepcopy(old)
    run = Run.random((2,3),nsetups=3)
    new *= run*.3
    new *= .3*run
    new /= .01*run
    new /= 9*run
    assert(np.isclose(new.values,old.values).all())

