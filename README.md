# KUKA iiwa Learning from Demonstration

## Installation

* Check that your computer's distribution of Python is 3.5.2 or above. If running on an older distribution of Linux (e.g Ubuntu 16), you will want to build a newer distribution of Python from source.
* Set up a virtual environment, e.g `python3.5 -m venv lfd_env3.5`. Refer to [venv documentation](https://docs.python.org/3/tutorial/venv.html) for more detail.
* First install and set up the `pypbdlib/` folder, install dependencies via `pip install -e .` in the top level of the directory.
* To test the LfD method, first run `python path.py` to test the program. Install any dependencies outlined by error messages. `pip install PACKAGE_NAME` should resolve any missing package errors.
### Dependencies
* numpy
* scipy
* matplotlib
* scikit-learn
* seaborn
* pandas
* pypbdlib (see main README for installation)

## Instructions

* `path.py` is the main script to use for normal testings.
* `path_transfer.py` is used for transfer tests.
* `check_path.py` is primarily used for testing out features and debugging.

* Information entropy is calculated in `normalize()` in `path.py`. Parameters are normalized and weighted, which is then used to calculate the entropy of a given point. The max ent region is then selected for the next demonstration.
* Editable hyperparameters are found in the `Params` dataclass. Default values are consistent with those used in the reported experiments. `vis_mode` is used to toggle between the entropy method and the 4cm rule baseline.
* The overall active learning loop is as follows:
  1. Start with initial demonstrations
  2. Fit the TP-GMM model
  3. Generate grid of potential trajectories
  4. Calculate information entropy for each region
  5. Select the point with maximum entropy
  6. Add/requeset a new demonstration at this point
  7. Repeat steps 2-6 until full state space coverage achieved
 
## Citation
Thank you for using and referencing our code. If using our library in your own research, please cite:
[CITATION NEEDED]
