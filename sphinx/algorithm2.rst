Algorithm 2: matchTemplate with meanShift
==========================================

How to run:

1. Install dependencies and requirements
2. Select a set of images (at least 100 images) with good lighting condition distributed during the whole period, and copy them into one folder. This calibration set of images will be used to calculate an appropriate conversion factor (to convert displacements from pixel to meter)
3. Select a set of templates (about a dozen) as explained in https://rtgmc.readthedocs.io/en/latest/algorithm1.html#important-instructions-for-the-template-image and save them into one folder
4. Import the script and run the main script, i.e. mT_mS.py with following inputs:

* path: select the path to the folder with image time series (see https://rtgmc.readthedocs.io/en/latest/algorithm1.html#important-instructions-for-the-image-series)
* path_cal: select the path to folder with calibration set of images (step 2 from above)
* path_template: select the path to the folder with templates (step 3 from above)

The workflow of the algorithm is as follow:

1. In the main script (mT_mS.py) the image time series is imported with the following function:

.. automodule:: mT_mS
    :members: load_good_images_from_folder

2. The parameters (a, b) of the function used to convert displacements into metric unit are calculated from the images in
the calibration set (good lighting conditions) with the following function:

.. automodule:: mT_mS
    :members: find_conversion_factor

3. The combination of following functions allow to calculate displacements of the pole with tapes for the time series
of images.

Main Functions
---------------
Considering two consecutive images, the `match_template <https://rtgmc.readthedocs.io/en/latest/algorithm2.html#mT.match_template>`_
function finds the initial location of the tapes in the first image. The function `meanShift <https://docs.opencv.org/master/d7/d00/tutorial_meanshift.html>`_
is than able to track the tapes in the consecutive image. This combination is implemented in the function `mS_different_frames <https://rtgmc.readthedocs.io/en/latest/algorithm2.html#mS.mS_different_frames>`_.
During the implementation of the algorithm, errors arising from a template that was not perfectly centered were observed.
In addition, since the pole and therefore the tapes may tilt over time, it is possible that a tape is centered at the
beginning of the time series but loses its centering over time, thus leading to erroneous results.
To overcome this problem, the function `mS_same_frame <https://rtgmc.readthedocs.io/en/latest/algorithm2.html#mS.mS_same_frame>`_
recalls `match_template <https://rtgmc.readthedocs.io/en/latest/algorithm2.html#mT.match_template>`_ and `meanShift <https://docs.opencv.org/master/d7/d00/tutorial_meanshift.html>`_
on the same frame, thus correcting possible errors from template offsets.

.. automodule:: mT
    :members: match_template

.. automodule:: mS
    :members: mS_different_frames, mS_same_frame

Sub-functions
---------------------------------------
The following functions are recalled by the `match_template <https://rtgmc.readthedocs.io/en/latest/algorithm2.html#mT.match_template>`_
function to remove duplicates matches and find collinear matches, i.e. on one straight line.

.. automodule:: mT
    :members: find_collinear, remove_duplicates

Since this algorithm works with colors to track tapes, the following functions are used to mask tape colors.

.. automodule:: mask
    :members: yellow, red, blue, green, black

Recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Run the algorithm with at least 10 different templates (template is a sensitive variable)
* If possible build the stations in such a way that lighting condition are goods, i.e. colors are well recognizable and the contrast is not too high
* By comparing the results of different templates, as well as results of Algorithm 1, some erroneous results may be filtered out thus obtaining better performances