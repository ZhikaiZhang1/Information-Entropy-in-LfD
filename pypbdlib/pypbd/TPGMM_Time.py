import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from scipy import linalg


def get_frame_dtype(dim):
    """
    :brief: Returns the numpy dtype for demonstration frames
    :param: Dimension of data (e.g. 3 for xyz position, 2 for xy position in 2D space)
    """
    return np.dtype([("A", "f8", (dim, dim)), ("b", "f8", (dim,))])


def GMM_time_initialize(data, n_states, reg_covar):
    """
    Provides initial GMM parameters, for input with an expected dependance on time.

    data should be a (n_features + 1, n_samples) array, with index 0 in the first dimension corresponding to time.
    n_states indicates the number of states to initialize for
    reg_covar is a regularization term.

    Returns tuple (priors, Mu, Sigma)
    priors is an array of size (n_states) indicating the weight of each component
    Mu is an array of size (n_features + 1, n_states) indicating the means.
    Sigma is an array of size (n_features + 1, n_features + 1, n_states) indicating the covariances.
    """
    priors = np.empty(n_states)
    n_var = data.shape[0]
    TimingSep = np.linspace(np.min(data[0, :]), np.max(data[0, :]), n_states + 1)

    Mu = np.zeros((n_var, n_states))
    Sigma = np.zeros((n_var, n_var, n_states))
    for i in range(n_states):
        idtmp = np.logical_and(
            data[0, :] >= TimingSep[i], data[0, :] < TimingSep[i + 1]
        )
        Mu[:, i] = np.mean(data[:, idtmp], 1)
        Sigma[:, :, i] = np.cov(data[:, idtmp]) + np.eye(n_var) * reg_covar
        priors[i] = len(idtmp)
    priors = priors / np.sum(priors)  # normalize
    return priors, Mu, Sigma


class TPGMM_Time:
    """
    Common symbols used:
    `T` is # of time steps in each demo
    `D` is # of task space dimensions
    `N` is # of demonstrations
    `F` is # of frames
    """

    def __init__(
        self, n_states, *, min_steps=5, max_steps=100, reg_covar=1e-5, tol=1e-5
    ):
        """
        Required parameters:
        n_states (number of gaussian components)

        Allowed optional parameters:
        min_steps (min # of EM steps)
        max_steps (max # of EM steps)
        reg_covar (diagonal regularization factor)
        tol (EM converges below this threshold)
        """
        self._n_states = n_states
        self._min_steps = min_steps  # nbMinSteps
        self._max_steps = max_steps  # nbMaxSteps
        self._reg_covar = reg_covar  # diagRegFact
        self._tol = tol  # maxDiffLL

        self._priors = None
        self._Mu = None
        self._Sigma = None
        # self._time = None
        self._n_parameters = None
        self._data_all = None

    def _transform_data(self, data_global, frames):
        """
        Does the heavy lifting for transforming the global data into each task parameter's
        position frames.

        data_global: shape (D+1, TxN)

        Returns array of size (D+1, F, T * N), holding demonstrations as observed from each frame
        """
        T, D, N, F = self._T, self._D, self._N, self._F
        D_plus_t = D + 1

        dat_all = np.empty((D_plus_t, F, T * N))
        data_global_x = data_global[1:]

        dat_all[0] = data_global[0]  # time
        print("number of Frame is: ", F)
        for f in range(F):
            for n in range(N):
                tslice = slice(n * T, (n + 1) * T)
                frame = frames[f, n]
                A = frame["A"]
                b = frame["b"]
                print("A is: " + str(A))
                print("B is: " + str(b))
                dat_all[1:, f, tslice] = np.linalg.solve(
                    A, data_global_x[:, tslice] - b[:, np.newaxis]
                )
        return dat_all

    def _initial_guess(self, dat_all):
        """
        Initializes the GMM based on time; i.e. first guess is that Gaussian components are spaced evenly in time.
        dat_all is size ((D+1), F, T x N) array holding demonstrations observed from each frame.

        Returns tuple (priors, Mu, Sigma). priors is a vector with length equal to # states.

        """
        T, D, N, F = self._T, self._D, self._N, self._F
        dat_flat = dat_all.reshape(((D + 1) * F, T * N), order="F")

        priors, Mu_flat, Sigma_flat = GMM_time_initialize(
            dat_flat, self._n_states, self._reg_covar
        )

        # Reshape GMM parameters into a 3-D array
        Mu = Mu_flat.reshape(((D + 1), F, self._n_states), order="F")
        Sigma = np.empty(((D + 1), (D + 1), F, self._n_states))
        for f in range(F):
            # Difficult to vectorize directly since we skip off-diagonal blocks
            frame_slice = slice(f * (D + 1), (f + 1) * (D + 1))
            Sigma[:, :, f] = Sigma_flat[frame_slice, frame_slice]
        return priors, Mu, Sigma

    def _computeGamma(self, Data, priors, Mu, Sigma):
        T, D, N, F = self._T, self._D, self._N, self._F

        GAMMA0 = np.zeros((self._n_states, F, T * N))
        for i in range(self._n_states):
            for f in range(F):
                GAMMA0[i, f, :] = multivariate_normal.pdf(
                    Data[:, f, :].T,
                    Mu[:, f, i],
                    Sigma[:, :, f, i],
                    allow_singular=True,
                )
        Lik = np.prod(GAMMA0, axis=1) * priors[:, np.newaxis]
        GAMMA = Lik / (np.sum(Lik, 0) + np.finfo(float).tiny)
        return Lik, GAMMA, GAMMA0

    def _GMM_EM(self, data, priors, Mu, Sigma):
        """
        data of shape ((D+1), F, T x N)
        """
        T, D, N, F = self._T, self._D, self._N, self._F
        D_plus_t = D + 1

        lastLL = -np.Inf
        for nbIter in range(self._max_steps):
            # E-step
            L, GAMMA, GAMMA0 = self._computeGamma(data, priors, Mu, Sigma)
            GAMMA2 = GAMMA / np.sum(GAMMA, 1)[:, np.newaxis]
            # Pix = GAMMA2

            # M-step
            # TODO - see if this can be done through broadcasting
            # TODO - document temp matrix sizes
            priors = np.sum(GAMMA, axis=1) / (T * N)
            for i in range(self._n_states):
                for f in range(F):
                    DataMat = data[:, f, :]  # Matricization/flattening of tensor
                    Mu[:, f, i] = DataMat @ GAMMA2[i]

                    DataTmp = DataMat - Mu[:, f, i][:, np.newaxis]
                    Sigma[:, :, f, i] = (
                        DataTmp @ np.diag(GAMMA2[i, :]) @ DataTmp.T
                        + np.eye(DataTmp.shape[0]) * self._reg_covar
                    )
            thisLL = np.sum(np.log(np.sum(L, axis=0))) / L.shape[1]
            if nbIter > self._min_steps and thisLL - lastLL < self._tol:
                self.n_iter_ = nbIter
                self.message = "EM converged after {} iterations.".format(self.n_iter_)
                break
            lastLL = thisLL
        else:
            self.n_iter_ = nbIter
            self.message = (
                f"The maximum number of {nbIter} EM iterations has been reached."
            )
        '''
        for i in range(Sigma.shape[0]):
            for j in range(Sigma.shape[1]):
                for k in range(Sigma.shape[2]):
                    for l in range(Sigma.shape[3]):
                        if Sigma[i,j,k,l] < 0.02:
                            Sigma[i,j,k,l] = 0.02
        '''

        return priors, Mu, Sigma, GAMMA0, GAMMA2

    def fit(self, data, frames, quiet=False):
        """
        :brief: Fit TPGMM to model. Assumes frames are constant in time.

        :param data: size (T, D+1, N) array, containing collected demonstration data (in global frame).
        First dimension indexes into time, second dimension indexes into the dimension of data, and the third dimension indexes into which demonstration to use.
        Note that the second dimension is of size D+1 despite there being only D dimensions; this is because index 0 in 2nd dimension is for time; remainder are for global position. It may be assumed in some methods that times are evenly spaced and same across different demonstrations; in any case this is recommended so that the training data is unbiased/uniform.

        :param frames: Array of shape (F, N) of frames (dtype given by get_frame_dtype(D)).

        Returns the fitted TP-GMM model (tuple of Mu and Sigma) - you can return the resulting object to load_model to skip fitting if it's already been done, saving you some computation time if you need to frequently fit data.
        """
        T, D_plus_t, N = data.shape
        D = D_plus_t - 1
        F, _ = frames.shape
        assert frames.shape[1] == N

        self._T, self._D, self._N, self._F = T, D, N, F  # save

        # Flatten data to (D+1, T x N) array (demos get concatenated)
        data_global = np.swapaxes(data, 1, 2).reshape((T * N, D + 1), order="F").T

        # data_flat is ((D+1) x F, T x N) array. (Used to be DataAll in init_TPGMM_timebased)
        # data_all is ((D+1), F, T x N) array. (Used to be Data)
        # TODO document size of priors, Mu, and Sigma
        data_all = self._transform_data(data_global, frames)
        self._data_all = data_all

        priors, Mu, Sigma = self._initial_guess(data_all)
        self._priors, self._Mu, self._Sigma, _, _ = self._GMM_EM(
            data_all, priors, Mu, Sigma
        )
        print("smallest variance is: ", np.min(abs(Sigma)))
        #print("smallest variance 2 is: ", np.min((Sigma)))
        '''
        for i in range(Sigma.shape[0]):
            for j in range(Sigma.shape[1]):
                for k in range(Sigma.shape[2]):
                    for l in range(Sigma.shape[3]):
                        if abs(Sigma[i,j,k,l])< 0.001:
                            if Sigma[i, j, k, l] < 0:
                                Sigma[i, j, k, l] = -0.001
                            else:
                                Sigma[i, j, k, l] = 0.001
        '''
        '''
        min_diag = 10
        min_off_diag = 10

        for i in range(Sigma.shape[3]):
            for j in range(Sigma.shape[2]):
                for k in range(Sigma.shape[1]):
                    for l in range(Sigma.shape[0]):
                        if l != k:
                            if abs(min_off_diag) > abs(Sigma[l, k, j, i]):
                                min_off_diag = Sigma[l, k, j, i]
                        elif l == k:
                            if abs(min_diag) > abs(Sigma[l, k, j, i]):
                                min_diag = Sigma[l, k, j, i]
                        #if l != k and abs(Sigma[l, k, j, i]) < 0.0001:
                        #    if Sigma[l, k, j, i] < 0:
                        #        Sigma[l, k, j, i] = -0.0001
                        #    else:
                        #        Sigma[l, k, j, i] = 0.0001
                        
                        #if l == k and abs(Sigma[l, k, j, i]) < 0.0001:
                        #    if Sigma[l, k, j, i] < 0:
                        #        Sigma[l, k, j, i] = -0.0001
                        #    else:
                        #        Sigma[l, k, j, i] = 0.0001

                #print("sigma here is: ", Sigma[:,:,j,i])
                #print("diagonal is: ", [Sigma[0,0,j,i], Sigma[1,1,j,i], Sigma[2,2,j,i]])
            
        print ("min diagnonal variance is: ", min_diag)
        print("min off-diagnonal variance is: ", min_off_diag)
        '''


        #("sigmaaaaa is: ", (Sigma))
        if not quiet:
            print(self.message)
        return self._priors, self._Mu, self._Sigma

    def load_model(self, model):
        """
        Passing in the saved object returned from fit() loads in the fitted model, without needing to go through the EM algorithm again.
        """
        self._priors, self._Mu, self._Sigma = model
        self._D, self._F, self._n_states = self._Mu.shape
        self._D -= 1  # get rid of time


    def bic_test(self, X, times):
        in_ = slice(0, 1)
        #out = slice(1, self._D + 1)
        #mixture = np.zeros((self._n_states, self._F))
        mixture = np.zeros((len(X),self._n_states))
        #print("data_all shape is: ", self._data_all.shape)

        for m in range(self._F):
            # Compute activation weights
            print("SHAPE: priors, Mu, Sigma, X", self._priors, self._Mu.shape, self._Sigma.shape, X.shape)
            for i in range(self._n_states):
                mixture[:, i] += self._priors[i] * multivariate_normal.pdf(
                    np.transpose(self._data_all[:,m,:]), self._Mu[:, m, i], self._Sigma[:, :, m, i]
                )
        K = self._Mu.shape[0]-1
        D = X.shape[1]
        N = X.shape[0]
        print("N is: ", N)
        npkd = K-1+K * D+K*D*(D+1)/2

        loglike = np.sum(np.log(np.sum(mixture, axis=1)))
        print("loglike is: ", loglike)
        score = -loglike + (npkd) * np.log(N)

        #print("mixture is: " + str(np.sum(mixture, axis = 0)))
        return score

    def reproduce(self, frames, times):
        """
        :param frames: Array of size (F, ) of frames (dtype given by get_frame_dtype(D)), representing the task parameters of the reproduced situation.
        times: Array of times at which the returned trajectory is specified; usually should be the same as times specified for demonstrations.

        Returns: reproduction, of dtype [("t", "f8", (T,)), ("x", "f8", (D, T)), ("cov", "f8", (D, D, T))].
        "t" holds the time (copied in from `times`) parameter
        "x" holds the reproduced trajectory
        "cov" holds the covariance at each point
        """
        # TODO make times optional by storing default during fitting
        # TODO extend so multiple repros can be done at once

        assert self._Mu is not None
        T, D, F = len(times), self._D, self._F

        in_ = slice(0, 1)
        out = slice(1, self._D + 1)

        # Array holding reproduction
        repro_dtype = np.dtype(
            [("t", "f8", (T,)), ("x", "f8", (D, T)), ("cov", "f8", (D, D, T))]
        )
        rnews = np.empty(1, dtype=repro_dtype)
        rnew = rnews[0]
        rnew["t"] = times

        # Variables for the gaussian component after GMR in each frame
        MuGMR = np.zeros((D, T, F))
        SigmaGMR = np.zeros((D, D, T, F))

        H = np.empty((self._n_states, T))  # activation weights
        for m in range(F):
            # Compute activation weights
            for i in range(self._n_states):
                H[i, :] = self._priors[i] * multivariate_normal.pdf(
                    times, self._Mu[in_, m, i], self._Sigma[in_, in_, m, i]
                )

            H /= np.sum(H, 0)
            for t in range(T):
                for i in range(self._n_states):
                    # Compute conditional means
                    MuTmp = self._Mu[out, m, i] + np.linalg.solve(
                        self._Sigma[in_, in_, m, i].T, self._Sigma[out, in_, m, i].T
                    ) * (times[t] - self._Mu[in_, m, i])
                    MuTmp = MuTmp.T
                    MuGMR[:, t, m] += np.squeeze(H[i, t] * MuTmp)

                    # Compute conditional covariances
                    SigmaTmp = (
                        self._Sigma[out, out, m, i]
                        - np.linalg.solve(
                            self._Sigma[in_, in_, m, i].T, self._Sigma[out, in_, m, i].T
                        ).T
                        @ self._Sigma[in_, out, m, i]
                    )
                    SigmaGMR[:, :, t, m] += H[i, t] * (SigmaTmp + MuTmp @ MuTmp.T)
                SigmaGMR[:, :, t, m] = (
                    SigmaGMR[:, :, t, m]
                    - np.outer(MuGMR[:, t, m], MuGMR[:, t, m])
                    + np.eye(D) * self._reg_covar
                )

        # model.nbVar = D + 1
        MuTmp = np.empty((D, T, F))
        SigmaTmp = np.empty((D, D, T, F))
        #print("F is: " + str(F))
        for m in range(F):
            # Linear transformation of the retrieved Gaussians
            MuTmp[:, :, m] = (
                frames[m]["A"] @ MuGMR[:, :, m] + frames[m]["b"][:, np.newaxis]
            )
            for t in range(T):
                SigmaTmp[:, :, t, m] = (
                    frames[m]["A"] @ SigmaGMR[:, :, t, m] @ frames[m]["A"].T
                )

        # Product of Gaussians (fusion of information from the different coordinate systems)
        for t in range(T):
            SigmaP = np.zeros((D, D))
            MuP = np.zeros((D))
            for m in range(F):
                SigmaP += np.linalg.inv(SigmaTmp[:, :, t, m])
                MuP += np.linalg.solve(SigmaTmp[:, :, t, m], MuTmp[:, t, m])

            rnew["cov"][:, :, t] = np.linalg.inv(SigmaP)
            rnew["x"][:, t] = rnew["cov"][:, :, t] @ MuP
        min_off_diag = 100
        min_diag = 100
        for j in range(SigmaTmp.shape[3]):
            for i in range(SigmaTmp.shape[2]):
                for k in range(SigmaTmp.shape[1]):
                    for l in range(SigmaTmp.shape[0]):
                        if l != k:
                            if abs(min_off_diag) > abs(SigmaTmp[l, k, i, j]):
                                min_off_diag = SigmaTmp[l, k, i, j]
                        elif l == k:
                            if abs(min_diag) > abs(SigmaTmp[l, k, i, j]):
                                min_diag = SigmaTmp[l, k, i, j]
        print("after mult min diagnonal variance is: ", min_diag)
        print("after mult min off-diagnonal variance is: ", min_off_diag)
        return rnews, self._Mu, self._Sigma

    def bic(self, X, times):
        """Bayesian information criterion for the current model on the input X.
        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)
            The input samples.
        Returns
        -------
        bic : float
            The lower the better.
        """
        '''
        p = self.mixtures(times)
        K = self._Mu.shape[2]
        D = X.shape[0]
        N = X.shape[1]
        np = K*D

        print("p is: " + str(p))
        loglike = np.sum(np.log(np.sum(p, axis = 0)))
        score = loglike - (np / 2) * log(N)
        '''
        Sigma_params = self._n_states * (self._Sigma.shape[0]-1) * self._Sigma.shape[0] / 2.0
        Mu_params = self._Sigma.shape[0]-1 * self._n_states
        self._n_parameters = int(Sigma_params + Mu_params + self._n_states - 1)
        return -2 * self.score(X) * X.shape[0] + self._n_parameters * np.log(
            X.shape[0]
        )
        #return score

    def score(self, X, y=None):
        """Compute the per-sample average log-likelihood of the given data X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        log_likelihood : float
            Log-likelihood of `X` under the Gaussian mixture model.
        """
        return self.score_samples(X).mean()

    def score_samples(self, X):
        """Compute the log-likelihood of each sample.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        Returns
        -------
        log_prob : array, shape (n_samples,)
            Log-likelihood of each sample in `X` under the current model.
        """
        #check_is_fitted(self)
        #X = self._validate_data(X, reset=False)

        return logsumexp(self._estimate_weighted_log_prob(X), axis=1)

    def _estimate_weighted_log_prob(self, X):
        """Estimate the weighted log-probabilities, log P(X | Z) + log weights.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Returns
        -------
        weighted_log_prob : array, shape (n_samples, n_component)
        """
        return self._estimate_log_prob(X) + np.log(self._priors)


    def _estimate_log_prob(self, X):
        return self._estimate_log_gaussian_prob(
            X, self._Mu, self._compute_precision_cholesky(
                self._Sigma, 'full'
            ),
            'full'
        )

    def _compute_precision_cholesky(self, covariances, covariance_type):
        """Compute the Cholesky decomposition of the precisions.

        Parameters
        ----------
        covariances : array-like
            The covariance matrix of the current components.
            The shape depends of the covariance_type.

        covariance_type : {'full', 'tied', 'diag', 'spherical'}
            The type of precision matrices.

        Returns
        -------
        precisions_cholesky : array-like
            The cholesky decomposition of sample precisions of the current
            components. The shape depends of the covariance_type.
        """
        estimate_precision_error_message = (
            "Fitting the mixture model failed because some components have "
            "ill-defined empirical covariance (for instance caused by singleton "
            "or collapsed samples). Try to decrease the number of components, "
            "or increase reg_covar."
        )
        covariances_fix = np.mean(covariances, axis=2)
        covariances_fix = np.transpose(covariances_fix, (2, 0, 1))
        if covariance_type == "full":
            print("sigma shape is: ", covariances.shape)

            n_components = covariances.shape[3]
            n_features = covariances.shape[1]
            precisions_chol = np.empty((n_components, n_features, n_features))
            #covariances_fix = np.transpose(covariances, (3,0,1,2))

            #print("covariances: " + str(covariances))

            for k, covariance in enumerate(covariances_fix):
                #print("covariance matrix now: ", covariances_fix)
                #print("k matrix shape", k)
                try:
                    cov_chol = linalg.cholesky(covariance)
                except linalg.LinAlgError:
                    raise ValueError(estimate_precision_error_message)
                precisions_chol[k] = linalg.solve_triangular(
                    cov_chol, np.eye(n_features), lower=True
                ).T
        elif covariance_type == "tied":
            n_features = covariances.shape[1]
            try:
                cov_chol = linalg.cholesky(covariances_fix, lower=True)
            except linalg.LinAlgError:
                raise ValueError(estimate_precision_error_message)
            precisions_chol = linalg.solve_triangular(
                cov_chol, np.eye(n_features), lower=True
            ).T
        else:
            if np.any(np.less_equal(covariances_fix, 0.0)):
                raise ValueError(estimate_precision_error_message)
            precisions_chol = 1.0 / np.sqrt(covariances_fix)
        return precisions_chol


    def _estimate_log_gaussian_prob(self, X, means, precisions_chol, covariance_type):
        """Estimate the log Gaussian probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        means : array-like of shape (n_components, n_features)

        precisions_chol : array-like
            Cholesky decompositions of the precision matrices.
            'full' : shape of (n_components, n_features, n_features)
            'tied' : shape of (n_features, n_features)
            'diag' : shape of (n_components, n_features)
            'spherical' : shape of (n_components,)

        covariance_type : {'full', 'tied', 'diag', 'spherical'}

        Returns
        -------
        log_prob : array, shape (n_samples, n_components)
        """
        print("X shape is: " + str(X.shape) + " mean shape is: " + str(means.shape))
        # average out means for all the frames, and also do transpose
        #sort out X's shape
        n_samples, n_features = X.shape
        #n_samples = X.shape[0]
        #n_features = X.shape[1] - 1
        #n_components, _ = means.shape
        n_components = means.shape[2]
        mean_average = np.mean(means, axis=1)
        mean_average = np.transpose(mean_average)
        # det(precision_chol) is half of det(precision)
        log_det = self._compute_log_det_cholesky(precisions_chol, covariance_type, n_features)

        if covariance_type == "full":
            log_prob = np.empty((n_samples, n_components))
            for k, (mu, prec_chol) in enumerate(zip(mean_average, precisions_chol)):
                #print("first dot: " + str(np.dot(X, prec_chol).shape) + " second dot: " + str(np.dot(mu, prec_chol).shape))
                y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
                #print("y is: " + str(y.shape) + " log_prob is: " + str(log_prob.shape))
                log_prob[:, k] = np.sum(np.square(y), axis=1)

        elif covariance_type == "tied":
            log_prob = np.empty((n_samples, n_components))
            for k, mu in enumerate(mean_average):
                y = np.dot(X, precisions_chol) - np.dot(mu, precisions_chol)
                log_prob[:, k] = np.sum(np.square(y), axis=1)

        elif covariance_type == "diag":
            precisions = precisions_chol ** 2
            log_prob = (
                    np.sum((mean_average ** 2 * precisions), 1)
                    - 2.0 * np.dot(X, (mean_average * precisions).T)
                    + np.dot(X ** 2, precisions.T)
            )

        elif covariance_type == "spherical":
            precisions = precisions_chol ** 2
            log_prob = (
                    np.sum(mean_average ** 2, 1) * precisions
                    - 2 * np.dot(X, mean_average.T * precisions)
                    + np.outer(row_norms(X, squared=True), precisions)
            )
        return -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det

    def _compute_log_det_cholesky(self, matrix_chol, covariance_type, n_features):
        """Compute the log-det of the cholesky decomposition of matrices.

        Parameters
        ----------
        matrix_chol : array-like
            Cholesky decompositions of the matrices.
            'full' : shape of (n_components, n_features, n_features)
            'tied' : shape of (n_features, n_features)
            'diag' : shape of (n_components, n_features)
            'spherical' : shape of (n_components,)

        covariance_type : {'full', 'tied', 'diag', 'spherical'}

        n_features : int
            Number of features.

        Returns
        -------
        log_det_precision_chol : array-like of shape (n_components,)
            The determinant of the precision matrix for each component.
        """
        if covariance_type == "full":
            n_components, _, _ = matrix_chol.shape
            log_det_chol = np.sum(
                np.log(matrix_chol.reshape(n_components, -1)[:, :: n_features + 1]), 1
            )

        elif covariance_type == "tied":
            log_det_chol = np.sum(np.log(np.diag(matrix_chol)))

        elif covariance_type == "diag":
            log_det_chol = np.sum(np.log(matrix_chol), axis=1)

        else:
            log_det_chol = n_features * (np.log(matrix_chol))

        return log_det_chol