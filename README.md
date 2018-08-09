# CRISPY
# The Coronagraph Rapid Imaging Spectrograph in Python
Simulates the WFIRST IFS

Documentation can be found:
https://mjrfringes.github.io/crispy/index.html

Maxime Rizzo, Tim Brandt, Neil Zimmerman, Tyler Groff, Prabal Saxena, Mike McElwain, Avi Mandell

NASA Goddard Space Flight Center


To install (work in progress):
1) clone the repository
2) cd into the main directory and type python setup.py install
3) then you can run the notebooks (in docs/source/notebooks/)

Alternatively, you can run the notebooks by adding the git folder path to the path by:
```python
import sys
folder = '../../../../crispy'
if folder not in sys.path: sys.path.append(folder)
```

