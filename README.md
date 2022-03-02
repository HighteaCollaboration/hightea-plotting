# hightea-plotting

Plotting routines for `hightea` library.

Documentation is available [online](https://hightea-plotting.readthedocs.io/en/latest/).

Simplest usage in python code:
```
import hightea.plotting as hyt

hyt.plot('tests/input/simple1d.json')
```
Plot function can take any arguments with appropriate data format
and a number of key arguments to manage plot features.

Data storage is organised in Run class which can be imported directly through
`hyt.Run`.

Run class can be initialised through file (`xml`,`csv`,`json`) or dictionary.

To make a contribution, report a bug, or suggest improvements,
please open an issue or email us directly at hightea@hep.phy.cam.ac.uk.
