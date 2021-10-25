from math import isclose
import numpy as np
import src.hightea.plotting as hyt
from src.hightea.stripper import convert_to_Run, load_to_Run
from src.hightea._MeasurementTools  import MeasurementTools


def test_loading_xml_1d():
    """Simple checks on 1d observable from stripper xml"""
    mt = MeasurementTools()
    mt.loadxml('tests/input/test.xml')
    run = convert_to_Run(mt, 0)
    assert(run.bins[0] == [[-0.5,0.5]])
    assert(len(run.edges) == 1)
    assert(len(run.bins[0]) == 1)
    assert(isinstance(run.edges[0], np.ndarray))


def test_loading_xml_2d():
    """Simple checks on 2d observable from stripper xml"""
    mt = MeasurementTools()
    mt.loadxml('tests/input/test.xml')
    run = convert_to_Run(mt, obs=1)
    assert(len(run.edges) == 2)
    assert(len(run.bins[0]) == 2)
    assert(len(run.values.shape) == 2)
    assert(isinstance(run.edges[0], np.ndarray))


def test_easyloading_xml_1d():
    run = load_to_Run('tests/input/test.xml')
    assert(run.bins[0] == [[-0.5,0.5]])


def test_plot_xml_1d():
    run = load_to_Run('tests/input/test.xml')
    hyt.plot(run, show=False)

