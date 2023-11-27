# Cranial Suture Growth Model
This is a repository for the [Data-driven cranial suture growth model enables predicting phenotypes of craniosynostosis](https://github.com/cuMIP/cranialSutureGrowth).

This repository also provides example scripts to generate synthetic normative cranial bone surface meshes, and to simulate single suture craniosynostosis. 

The data folder contains the trained surue growth parameters (``sutureGrowthModel``) regional weights at each anchor (``weights``) described in the manuscript.


## Dependencies:
- [Python](python.org)
- [NumPy](https://numpy.org/install/)
- [SimpleITK](https://simpleitk.org/)
- [VTK](https://pypi.org/project/vtk/)
- [scikit-learn](https://scikit-learn.org/stable/)

    *Once Python is installed, each of these packages can be downloaded using [Python pip](https://pip.pypa.io/en/stable/installation/)*


## Using the code

### Quick summary

### Code example

```python
import Tools
import pickle
import SimpleITK as sitk
import vtk
import numpy as np
import os