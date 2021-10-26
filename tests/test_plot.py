import src.hightea.plotting as hyt
from src.hightea.stripper import convert_to_Run, load_to_Run

def test_plot():
    hyt.plot('tests/input/simple1d.json', show=False)

def test_ratioPlot():
    hyt.plot(load_to_Run('tests/input/ppWpjs_0_nnlo.xml'),
             load_to_Run('tests/input/ppWpjs_T_nnlo.xml'),
             show=False, ratio=0)

def test_hepdata():
    hyt.plot('tests/input/HEPData-ins1610623-v1-Table_7.csv', show=False)

