[![Documentation Status](https://readthedocs.org/projects/rtgmc/badge/?version=latest)](https://rtgmc.readthedocs.io/en/latest/?badge=latest)

# Determine Real-Time Glacier Mass Changes from Camera Images

This project aims to set up a framework that is able to process series of camera images and automatically calculate a mass balance from them. Specifically, coloured tapes on metal poles are to be detected and tracked using computer vision algorithms.
The basic framework consists of the open source library [openCV](https://opencv.org/) in Python. This project contains two independent algorithms based on the openCV funtions matchTemplate and meanShift.

## Requirements
```python
numpy~=1.19.2
opencv-python~=4.4.0.42
imutils~=0.5.3
matplotlib~=3.3.2
pyqtgraph~=0.11.0
PyQt5~=5.15.1
pandas~=1.1.3
scikit-learn~=0.23.2
XlsxWriter~=1.3.7
```

## How to run Algorithm 1: matchTemplate with histograms
1. Download or clone repository, install dependencies
2. Create blank .py file
3. Import required script and run function
```python
import mT_Hist
x, displacements, conversion_factors = mT_Hist.matchTemplate_hist("myfolder", "template.jpg", 0.70, wait=1, vis=False, plotting=False, csv=True)
```
4. Analyze output.csv or plot return values

## How to run Algorithm 2: matchTemplate with meanShift
1. Install dependencies and requirements
2. Select a set of images (at least 100 images) with good lighting condition distributed during the hole period,
and copy them into one folder.
3. Select a set of templates (about a dozen) and save them into one folder.
4. Import the script and run the main script, i.e. mT_mS.py with following inputs:

* path: select the path to the folder with image time series
* path_cal: select the path to folder with calibration set of images (step 2 from above)
* path_template: select the path to the folder with templates (step 3 from above)
## Documentation
The documentation is hosted here: https://rtgmc.readthedocs.io/en/latest/ 
[![Documentation Status](https://readthedocs.org/projects/rtgmc/badge/?version=latest)](https://rtgmc.readthedocs.io/en/latest/?badge=latest)


