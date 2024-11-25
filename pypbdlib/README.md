## Note
This is where I (Jesse) setup the ports of MATLAB PBD demos over to Python. I was originally planning to merge this upstream, but that's looking more and more far-fetched.

## Installation
Before continuing - it is strongly recommended to first setup a virtual environment - look up the Python `venv` module if you're unfamiliar with virtual environments. This will isolate the packages you work with on this project.

`git clone` this repository to a local folder.
There is a `setup.py` file, so `pip install -e .` should install all dependencies, assuming you're in the `pypbdlib` directory. Then you can just run `from pypbd import TPGMM_Time`, etc. In Python, objects importable directly from `pypbd` are listed in `__init__.py`.

### More details on virtual environments
With a virtual environment - it's essentially a way to isolate between different projects what packages you install; this makes it possible to work on two different projects that require different package versions or specific package dependencies, or use different Python interpreter versions.

Python has a built-in module to do it called `venv` - some Googling should lead you to a command to create a virtual environment.

Typically, instructions will tell you to activate a virtual environment by sourcing some script in a special folder; then your terminal prompt will show that a particular environment is active. You can then install packages, which will be isolated from any other projects you have.

Depending on your IDE, there may be an option to select which interpreter to use - a virtual environment will provide a binary executable for the Python interpreter used in the environment; e.g. check the `bin/` folder.


## Documentation
The main tool of interest is the `TPGMM_Time` class; this models a time-based TP-GMM with fixed frames. The constructor takes in some hyperparameters, and the `fit` method runs the actual EM algorithm. Then, the `reproduce` method takes in new frames (task parameters) and returns a result. For computational efficiency, it is possible to save and load models - the `fit` method returns an object which can be passed to `load_model`, which will recover the results of fitting.

Frames are defined by a [NumPy structured array](https://numpy.org/doc/stable/user/basics.rec.html) - for convenience, the `get_frame_dtype` function returns the NumPy `dtype` object describing a frame, with two components - one for `A` and one for `b`.

Under `repro_joints.py`, there is a helpful function `repro_joints` which is useful for reproducing in joint space. The overarching algorithm is to first reproduce in task space, using a TP-GMM. Additionally, based on the demonstration data, learn the null-space control policy. Then, use an iterative (least-squares) IK solver, as an explicit task space constraint. Finally, apply the learned null-space control policy, projecting it into the null space. In practice, applying the policy may slightly change the actual task space position (due to the linearization approximation made in the formulation of the null space), so the last two steps of IK and applying policy need a bit of iteration to converge.

A simple learned null space controller is provided by the `GMMNullspaceController` class, which models the joint states with a GMM then performs GMR to estimate the unconstrained control policy. This is a simple but limited approach, and a more sophisticated approach is described in the Constraint Consistent Learning (CCL) library [(link to paper)](https://arxiv.org/pdf/1807.04676v1.pdf). It's not yet (as of 2020-12-29) provided in this library, but an example implementation of how to use it are available in the automation scripts for the demonstration quality quantifying project (PR2 box button pushing).

Refer to the actual `.py` files and the docstrings under each method for detailed information on parameters and data types.

## Example
Example code in the example/ folder, based on sample data from PBDLib.

## References
If you find these codes useful for your research, please acknowledge the authors by citing:

Pignat, E. and Calinon, S. (2017). [Learning adaptive dressing assistance from human demonstration](http://doi.org/10.1016/j.robot.2017.03.017). Robotics and Autonomous Systems 93, 61-75.
