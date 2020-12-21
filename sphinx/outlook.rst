Known Issues and Ideas for Further Development
================================================

The following section describes some problems with the algorithms that have not been fully solved, as well as ideas for further development.

Data Gaps
---------

Both algorithms are unable to calculate the correct displacements if longer periods of data are missing.
The maximum displacement that can be measured between two consecutive images is 4 cm and corresponds to a complete phase shift of the tapes on the pole.
The algorithms are however capable of detecting such data gaps and returning a warning, so that missing displacements can be added by hand.
This could be solved in the future by detecting color sequences. If the the predefined color sequence is also implemented, displacements could even be backtracked for cases where tapes from the first image have already moved out of sight. There is already an inactive color detection function in the first algorithm, which however does not work reliably. It will be very challenging to implement reliable color detection because in some lighting conditions, the colors are not even detectable by human eye. Under no circumstances should this function interfere with the continuous displacement calculations, as these few, unavoidable errors in colour recognition reduce the amount of detected tapes and thus generally reduce the stability of both algorithms. Given that these data gaps occur very rarely and a manual correction is generally not a lot of effort and less error-prone, it must be asked whether such an effort is worthwhile.

Possible Improvements with Different Tape Colors
--------------------------------------------------

Some tapes are easier to detect than others. Especially white and gray tapes are problematic due to their low contrast to the background and no color saturation. Also tapes with texture filling (yellow with black stripes) are not ideal. Contrast and saturation are the most important characteristics the tapes should have. Therefore, the stability of both algorithms can be increased by replacing problematic colors with red, blue, yellow or green tapes. However, it must also be mentioned that this is a direct conflict with the time gap problem. If the number of unique colours in the sequence decreases, the possibilities to trace back displacements from time gaps also decrease.

Link to Glacier Mass Balance
-----------------------------

Strictly speaking, the algorithms measure a change in height of the glacier surface compared to a fixed pole. In order to convert this into a glacier mass balance in [m w.e.], the displacements therefore have to be multiplied by a factor :math:`\rho_{ice}/\rho_{water}`. This is further complicated when cases like station 1001 are considered, which are installed on a snow cover. The mass balance there could be approximated with a factor :math:`\rho_{snow}/\rho_{water}`. This however suggests that only snow is melting during the time when a snow cover is existent, which is not entirely true (see report). Also, the exact snow density and the transition from snow melt to ice melt remain unsolved. In contrast to already existing snow cover during installation, short-term changes in the mass balance due to summer snowfall cannot be taken into account at the moment because the camera does not move relative to the pole both when snow falls and when the newly deposited snow cover melts.
Furthermore, the current sign convention suggests that glacier melt equals positive displacements. All values must therefore be inverted, as the common known sign convention suggests that melt results in a negative mass balance.

Continuous Integration and Delivery
------------------------------------

Due to lack of time and knowledge, there was no possibility to implement proper test environments on gitLab. This was not very problematic for this work, as the development of two algorithms in general did not cause many conflicts. However, in further work, such an implementation could still prove useful.

Runtime and Performance
------------------------

The main objective of this work was to prove the feasibility of such a workflow to automatically calculate mass balances. The runtime of the algorithms played a subordinate role. There is still room for improvement, as some functions, such as finding collinear matches, have a runtime of :math:`O(n^2)`. For the time series obtained, this was absolutely unproblematic, but when much more data is added, it could play a role. Nevertheless, it must be said that this time saving potential is a fraction of the time gain of automation compared to hand measurements.

Changes in Time Intervals
---------------------------

Both algorithms were tested and optimised for time intervals of 20 minutes between two consecutive images, except at night. With a shorter interval, no problems should be expected. However, for longer time intervals, the chance of a loss of correlation increases. Under no circumstances should the time intervals be so long that the displacements between two images exceed 4 cm. Strictly speaking, 4 cm is already too much in case no tapes can be detected in an image.