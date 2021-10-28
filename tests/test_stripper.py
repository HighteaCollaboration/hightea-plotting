import pytest
from math import isclose
import numpy as np
import src.hightea.plotting as hyt
from src.hightea.stripper import convert_to_Run, load_to_Run
from src.hightea._MeasurementTools  import MeasurementTools


def test_loading_xml_1d():
    """Load 1d observable from stripper xml"""
    mt = MeasurementTools()
    mt.loadxml('tests/input/test.xml')
    run = convert_to_Run(mt, 0)
    assert(run.bins[0] == [[-0.5,0.5]])
    assert(len(run.edges) == 1)
    assert(len(run.bins[0]) == 1)
    assert(isinstance(run.edges[0], list))
    newrun = load_to_Run('tests/input/test.xml')
    assert(newrun.bins[0] == [[-0.5,0.5]])


def test_loading_xml_2d():
    """Load 2d observable from stripper xml"""
    mt = MeasurementTools()
    mt.loadxml('tests/input/test.xml')
    run = convert_to_Run(mt, obs=1)
    assert(len(run.edges) == 2)
    assert(len(run.bins[0]) == 2)
    assert(len(run.values.shape) == 2)
    assert(isinstance(run.edges[0], list))


def test_plot_xml_1d():
    """Trivial plotting"""
    run = load_to_Run('tests/input/test.xml')
    hyt.plot(run, show=False)

def test_xsec():
    """Xsection list and list of scales should have matching length"""
    run = load_to_Run('tests/input/test.xml')
    assert(run.xsec.shape[1] == run.values.shape[1])

def test_loadObservables():
    """Load observable by name"""
    filename = 'tests/input/test.xml'
    mt = MeasurementTools()
    mt.loadxml(filename)
    expected = mt.extractObservables(0)[0]
    run = load_to_Run(filename, obs='total cross section')
    assert(run.info.get('obs') == expected)
    with pytest.warns(UserWarning):
        run = load_to_Run(filename, obs='transverse momentum')
        assert(not(run.info.get('obs') == expected))
    with pytest.raises(Exception):
        run = load_to_Run(filename, obs='#@!')

