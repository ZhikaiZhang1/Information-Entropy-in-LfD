# KUKA iiwa LfD

## Installation

* Check that your computer's distribution of Python is 3.5.2 or above. If running on an older distribution of Linux (e.g Ubuntu 16), you will want to build a newer distribution of Python from source.
* Set up a virtual environment, e.g `python3.5 -m venv lfd_env3.5`. Refer to [venv documentation](https://docs.python.org/3/tutorial/venv.html) for more detail.
* Install the pypbdlib/ folder, refer to the README for installation instructions
* `python path.py` to test the program, install any dependencies outlined by error messages. `pip install PACKAGE_NAME` should take care of most of them

## Instructions

* `path.py` is the main script to use for normal testings, `path_transfer.py` is used for transfer tests, `check_path.py` is primarily used for testing out features and debugging. 
