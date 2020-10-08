import numpy as np
from functools import reduce

from shfl.model.recommender import Recommender


class SVDRecommender(Recommender):
    """
    Implementation of an SVD recommender using \
        [Recommender](../model/#recommender-class)

    # Arguments:
        f: integer. Dimension of the latent space.
        lam: float. Regularization parameter.
        g: float. Step size in gradient descent.
        sensitivity: float.
        epsilon: float.

    The data used to train and test the model is a numeric numpy array in which the first column specifies the
    client and the second one corresponds to the item.

    The guess for the rating a user $u$ gives to item $i$ is
    $$
    \\hat r_{ui} = \\mu_u + p_u\\cdot q_i\\,,
    $$
    where $\\mu_u$ is the mean rating given by the user and $p_u,\\, q_i$ are vectors in $\\mathbb R^f$ that we wish
    to estimate. This is done by minimizing the L2 regularized squared error function.
    $$
    J(p,q) = \\frac{1}{2}\\sum_{(i,u)\\in\\mathcal K}\\left ( e^2_{ui} +
    \\lambda ||q_ i||^2 + \\lambda||p_ i||^2\\right )\\,,
    $$
    with $e_{ui} = r_{ui} - \\mu_u - p_u\\cdot q_i$.

    The optimization algorithm is a combination of Alternating Least Squares and Gradient Descent together with a
    DP mechanism. Each client computes its profile as
    $$
    p_u = \\left ( \\lambda\\mathbb I + \\sum_{\\mathcal K_u} q_i\\otimes q_i \\right )^{-1}
    \\sum_{\\mathcal K_u}(r_{ui} - \\mu) q_i
    $$
    as well as the following gradients
    $$
    \\nabla q_{iu} = - e_{ui}\\,p_u + \\lambda q_i \\,.
    $$
    In order to preserve the privacy of the ratings, they add Laplace noise to the gradients,
    $$
    \\widehat\\nabla q_{iu} =  \\nabla q_{iu} + {\\rm Laplace}(b)\\,,
    $$
    which they send to the central server.

    The server adds up all the gradients and updates the items' profiles as
    $$
    q_i \\leftarrow  q_i - g \\sum_{u\\in \\mathcal I_i} \\widehat \\nabla q_{iu}\\,.
    $$

    This process is repeated until convergence is reached.
    """

    def __init__(self, f, lam, g, sensitivity, epsilon):
        super().__init__()
        self._f = f
        self._lam = lam
        self._g = g
        self._sensitivity = sensitivity
        self._epsilon = epsilon

        self._grad = None
        self._q_items = None
        self._profile = None
        self._mu = None

    def _check_no_new_items(self, data):
        items_in_catalog = set(self._q_items.keys())
        items_in_data = set(data[:, 1].tolist())

        if not items_in_data.issubset(items_in_catalog):
            raise AssertionError("The items in the data are not included in the catalog.")

    def train_recommender(self, data, labels):
        """
        Method that trains the model

        # Arguments:
            data: Data to train the model. Only includes the data of this client and every item must be in the catalog.
            labels: Label for each train element
        """
        self._check_no_new_items(data)

        self._mu = np.mean(labels)

        n_obs = len(data)
        q_items = self._q_items

        items_in_data = data[:, 1].tolist()
        q_items_in_data = np.array([q_items[k] for k in items_in_data])

        # USER LATENT FACTORS
        mat = reduce(np.add,
                     [np.outer(q_items_in_data[i], q_items_in_data[i]) for i in range(n_obs)])
        mat = self._lam * n_obs * np.identity(self._f) + mat
        vec = reduce(np.add,
                     [q_items_in_data[i] * (labels[i] - self._mu) for i in range(n_obs)])
        p = np.matmul(np.linalg.inv(mat), vec)

        self._profile = p

        # GRADIENTS
        grad = [self._g * ((labels[i] - self._mu - np.dot(p, q_items_in_data[i])) * p - self._lam * q_items_in_data[i])
                for i in range(n_obs)]

        self._grad = dict(zip(items_in_data, grad))

    def predict_recommender(self, data):
        """
        Predict labels for data. Only includes the data of this client and every item must be in the catalog.

        # Arguments:
            data: Data for predictions. Only includes the data of this client

        # Returns:
            predictions: Array with predictions for data
        """
        self._check_no_new_items(data)

        items_in_data = data[:, 1].tolist()
        q_items_in_data = np.array([self._q_items[k] for k in items_in_data])

        predictions = self._mu + np.array([np.dot(q_items_in_data[i], self._profile)
                                           for i in range(len(data))])
        return predictions

    def evaluate_recommender(self, data, labels):
        """
        This method must returns the root mean square error

        # Arguments:
            data: Data to be evaluated. Only includes the data of this client and every item must be in the catalog.
            labels: True values of data of this client
        """
        predictions = self.predict(data)

        if predictions.size == 0:
            rmse = 0
        else:
            rmse = np.sqrt(np.mean((predictions - labels) ** 2))
        return rmse

    def get_model_params(self):
        """
        Gets the params that define the model

        # Returns:
            params: Mean rating
        """
        return self._grad

    def set_model_params(self, q_items):
        """
        Update the params that define the model

        # Arguments:
            params: Parameter defining the model
        """
        self._q_items = q_items

    def performance_recommender(self, data, labels):
        """
        This method returns the root mean square error of the recommender.

        # Arguments:
            data: Data to be evaluated. Only includes the data of this client and every item must be in the catalog.
            labels: True values of data of this client
        """
        predictions = self.predict(data)

        if predictions.size == 0:
            rmse = 0
        else:
            rmse = np.sqrt(np.mean((predictions - labels) ** 2))
        return rmse
