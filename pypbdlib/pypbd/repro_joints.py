import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture

from pypbd.TPGMM_Time import GMM_time_initialize


def _get_nullspace(t, js, robot, null_control):  # time and joint states
    """
    Find the nullspace control component.
    """
    jacob = robot.get_jacobian(js)
    null_projector = np.eye(robot.Q) - np.linalg.pinv(jacob) @ jacob
    return null_projector @ null_control.get_policy(t, js)


def repro_joints(
    tpgmm,
    frame,
    time,
    robot,
    null_control,
    start,
    *,
    max_speed=None,
    verbose=False,
    limit_joints=False,
    max_ik_null_iter=1,
):

    """
    Reproduces trajectories in joint space.

    tpgmm is a TPGMM_Time model that has already been fitted/with loaded model parameters
    frame is the task space frames/task parameters to use
    time is the times to be reproduced on
    robot is an object with some methods and properties relevant to the robot - see below
    null_control is the nullspace controller / secondary controller - see below
    start is the starting joint state.
    max_speed is the maximum speed between joint states - updates are scaled downward if this is exceeded. Assumed in rad/s, and time in s. This is useful, either to enforce a maximum joint speed or to help smooth out the discontinuities introduced when your orientation representation "wraps around".
    limit_joints is a flag - if active, a call to `robot.limit_joints(joint_state)` will be made, which should return a joint state with limits applied. `np.clip` is particularly useful for implementing this method.

    Returns an array of size (robot.Q, len(time) + 1) of joint states.

    ----

    `robot` must have the following methods available:
        robot.IK(initial_joint, target)
            given an array `initial_joint` holding initial joint states and a `target` in task space, iteratively solves inverse kinematics. Ideally, the IK solutions should be continuous, so iterative methods are preferred. Returns the joint state.

        robot.FK(joint_states)
            Solves the forward kinematics problem, and returns a vector in task space.

        robot.get_jacobian(joint_states)
            Finds the (geometric) Jacobian, given joint states. This is needed to implement null space control.

        robot.is_close(x1, x2)
            Returns true if the two task space positions x1 and x2 are close; this is used for testing convergence. This might be more complex than just looking at the difference between the two state vectors, e.g. for quaternion representation, there's a different thing you need to do to get the rotation distance

    `robot` must have the property:
        robot.Q
            Number of joints in the robot.


    null_control represents the null space control policy, and must have the following methods available;
        null_control.get_policy(time, joint_state)
            Given the `time` (scalar) and `joint_state` (vector), returns the control policy. The return value will be added on to the inital joint state, so e.g. if the policy is to target a preferred position, this method should return the difference between that position and `joint_state`.
            See the GMMNullspaceController as an example.
    """

    def printq(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    xrepro = tpgmm.reproduce(frame, time)["x"]
    T = len(time)
    joints = np.empty((robot.Q, T + 1))
    joints[:, 0] = start

    printq("Reproducing")
    for ti, t in enumerate(time):
        printq(f"step {ti}")
        x_tgt = xrepro[:, ti]
        converged = False
        js = joints[:, ti]
        # js += _get_nullspace(t, js, robot, null_control)
        # js = robot.IK(js, x_tgt)

        for ikit in range(max_ik_null_iter):  # max iterations
            js += _get_nullspace(t, js, robot, null_control)
            js = robot.IK(js, x_tgt)
            if robot.is_close(robot.FK(js), x_tgt):
                break

        if max_speed is None:
            new_js = js
        else:
            update = js - joints[:, ti]
            try:
                max_val = np.max(np.abs(update)) / (time[ti + 1] - time[ti])
            except IndexError:
                max_val = np.max(np.abs(update)) / (time[ti] - time[ti - 1])

            if max_val > max_speed:
                printq(f"Max speed ({max_speed:.3f}) exceeded ({max_val:.3f})")
                update *= max_speed / max_val
            else:
                printq(f"Diff ok ({max_val:.3f})")
            new_js = joints[:, ti] + update

        if limit_joints:
            new_js = robot.limit_joints(new_js)
        joints[:, ti + 1] = new_js
    return joints


class GMMNullspaceController:
    """
    Simple nullspace controller - estimate the null space control policy with a GMM of all the joint states.
    """

    def __init__(self, js_data, nbStates=7):
        """
        js_data is shape (T, Q+1, N)
        nbStates is the number of Gaussian components
        """
        self.nbStates = nbStates

        Q_plus = js_data.shape[1]  # Q + 1
        js_data = js_data.swapaxes(1, 2).reshape([-1, Q_plus], order="F")

        priors, Mu0, Sigma0 = GMM_time_initialize(js_data.T, self.nbStates, 1e-5)
        Mu0 = Mu0.T
        Sigma0 = np.moveaxis(Sigma0, [0, 1, 2], [1, 2, 0])
        self.model = GaussianMixture(
            self.nbStates,
            max_iter=300,
            weights_init=priors,
            means_init=Mu0,
            precisions_init=np.linalg.inv(Sigma0),
        )
        # expects (T * N, Q + 1)
        self.model.fit(js_data)
        self.Priors = self.model.weights_
        self.Mu = self.model.means_.T
        self.Sigma = np.transpose(self.model.covariances_, axes=[1, 2, 0])

    def _GMR(self, DataIn, in_=None, out=None):
        """
        Variables:
            D - number of features/dimensions, not including time

        DataIn is the time, or more generally the regression variables (?).

        in_ is optional and is a slice indicating which variables are inputs (e.g. time), by default this is just 0
        out is the slice indicating which variables are outputs, by default it assumes the rest of the variables.

        """
        if in_ is None:
            in_ = 0
        if out is None:
            out = slice(1, self.Mu.shape[0])

        # nbData = 1
        nbVarOut = out.stop - out.start

        MuTmp = np.zeros((nbVarOut, self.nbStates))
        expData = np.zeros(nbVarOut)
        # expSigma = zeros(nbVarOut, nbVarOut)

        # Compute activation weights
        H = np.empty(self.nbStates)
        for i in range(self.nbStates):
            H[i] = self.Priors[i] * multivariate_normal.pdf(
                DataIn, self.Mu[in_, i], self.Sigma[in_, in_, i]
            )
        H /= np.sum(H)

        # Compute conditional means
        for i in range(self.nbStates):
            MuTmp[:, i] = (
                self.Mu[out, i]
                + (DataIn - self.Mu[in_, i])
                * self.Sigma[out, in_, i]
                / self.Sigma[in_, in_, i]
            )
            expData[:] += H[i] * MuTmp[:, i]

            # SigmaTmp = model.Sigma[out,out,i] - model.Sigma[out,in_,i] / model.Sigma[in_,in_,i] * model.Sigma[in_,out,i]
            # expSigma[:,:,t] += H[i] * (SigmaTmp + np.sum(MuTmp(:,i) ** 2))
        return expData

    def get_policy(self, t, js):
        """
        Returns the control policy, given the time t and current joint state js.
        """
        expect = self._GMR(t)
        return expect - js
