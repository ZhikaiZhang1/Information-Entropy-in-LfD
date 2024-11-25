# KUKA iiwa LfD

## Installation

* Check that your computer's distribution of Python is 3.5.2 or above. If running on an older distribution of Linux (e.g Ubuntu 16), you will want to build a newer distribution of Python from source.
* Set up a virtual environment, e.g `python3.5 -m venv lfd_env3.5`. Refer to [venv documentation](https://docs.python.org/3/tutorial/venv.html) for more detail.
* Install [pypbdlib](https://gitlab.com/jesse.li2002/pypbdlib), refer to the README for installation instructions
* `python path.py` to test the program, install any dependencies outlined by error messages. `pip install PACKAGE_NAME` should take care of most of them

## Instructions

* `path.py` is the main script to use, `check_path.py` is primarily used for testing out features and debugging. 
* Refer to this [document](https://docs.google.com/document/d/1Hoe-KGAe_jLRom8S-55e_vHk4I268x6u/edit) for further instructions, including those regarding setting up the Linux workspace side of the project