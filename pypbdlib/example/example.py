# Sample data taken from PBDlib developed by Sylvain Calinon.
# Make sure that installation instructions for pypbdlib have been done first.
# You may also need to install `matplotlib` to run this example (plotting)

# Resulting plot shows four demonstrations (in black) moving from the purple
# U-shaped "peg" to yellow pegs, scattered in different locations. Using
# these demonstrations, a TPGMM model is trained, and a reproduction is done
# with a new set of task parameters; the repro is shown in green and the new
# target in cyan.

import pypbd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Some plotting functions
COL_PEGS_DEFAULT = ((0.2863, 0.0392, 0.2392), (0.9137, 0.4980, 0.0078))  # colours


def plotPegs(axis, ps, colPegs=COL_PEGS_DEFAULT, fa=0.6):
    """
    axis: matplotlib.pyplot.Axes
    ps: size (F=2,) array of frames
    """
    pegMesh = (
        np.array(
            (
                (-4, -3.5),
                (-4, 10),
                (-1.5, 10),
                (-1.5, -1),
                (1.5, -1),
                (1.5, 10),
                (4, 10),
                (4, -3.5),
                (-4, -3.5),
            )
        ).T
        * 1e-1
    )
    for p, colour in zip(ps, colPegs):
        dispMesh = p["A"][:2, :2] @ pegMesh + p["b"][:2, np.newaxis]
        axis.add_patch(patches.Polygon(dispMesh.T, fc=colour, alpha=fa))


def plotGMM(axis, Mu, Sigma, color, valAlpha=1.0):
    """
    :param Mu: D x K array representing the centers of K Gaussians.
    :param Sigma: D x D x K array representing the covariance matrices of K Gaussians.
    :param color: Length 3 sequence representing the RGB color to use for the display.
    :param valAlpha: float, transparency factor (default 1).
    """

    nbStates = Mu.shape[1]
    nbDrawingSeg = 100
    darkcolor = [0.7 * i for i in color]
    # t = linspace(-pi, pi, nbDrawingSeg)

    X = np.empty((2, nbDrawingSeg, nbStates))
    Sigmod = np.moveaxis(Sigma, 2, 0)
    w, v = np.linalg.eig(Sigmod)
    w = np.sqrt(w) * 2  # Need diameter
    widths = w[:, 0]
    heights = w[:, 1]
    vecs = v[:, :, 0]  # K x 2 array
    angles = np.arctan2(vecs[:, 1], vecs[:, 0]) * 180 / np.pi

    for i in range(nbStates):
        ellip = patches.Ellipse(
            Mu[:, i],
            widths[i],
            heights[i],
            angles[i],
            alpha=valAlpha,
            fc=darkcolor,
            ec=color,
        )
        axis.add_patch(ellip)
    axis.scatter(Mu[0, :], Mu[1, :], c=[darkcolor], s=10)


# Load in sample training data
dat_path = Path(__file__).parent / "data" / "data0.npz"
with np.load(dat_path) as loadin:
    data = loadin["data"]  # (T=200, D+1 = 3, N=4)
    frames = loadin["frames"]  # (F=2, N=4)
T, D, N = data.shape  # time, dimension of data, num repros
D -= 1  # recall that the first value is for time
F, _ = frames.shape


# Plot training data
fig, axes = plt.subplots()
axes.set_aspect("equal")
for n in range(4):
    axes.plot(data[:, 1, n], data[:, 2, n], color="black")
    plotPegs(axes, frames[:, n])

# Train model
tpgmm = pypbd.TPGMM_Time(3)
tpgmm.fit(data, frames)


# Create new task parameters for generalization
# (mish-mash of demonstration frames; this can be modified)
frame_dtype = frames.dtype
new_frame = np.empty(F, dtype=frame_dtype)
new_frame["A"] = frames[:, 0]["A"]
new_frame[0]["b"] = frames[0, 1]["b"]
new_frame[1]["b"] = (frames[1, 1]["b"] + frames[1, 2]["b"]) / 2

# Reproduce with model on with new task parameters
times = data[:, 0, 0]
repro = tpgmm.reproduce(new_frame, times)

# Plot reproduction and show results
axes.plot(repro["x"][0], repro["x"][1], color="green")
plotPegs(axes, new_frame, colPegs=[COL_PEGS_DEFAULT[0], (0, 1, 1)])

plt.show()
