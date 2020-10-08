import numpy as np
import pytest
from functools import reduce

from shfl.model.svd_recommender import SVDRecommender


def test_svd_recommender_train_wrong_data():
    f = 2
    lam = 0.1
    g = 0.01
    sensitivity = None
    epsilon = None

    q_items = {1: np.array([1, 7]),
               2: np.array([4, 6])}
    data = np.array([[1, 1],
                     [1, 5]])
    labels = np.array([1, 6])

    svd_recommender = SVDRecommender(f, lam, g, sensitivity, epsilon)
    svd_recommender._q_items = q_items

    with pytest.raises(AssertionError):
        svd_recommender.train(data, labels)


def test_svd_recommender_train():
    f = 2
    lam = 0.1
    g = 0.01
    sensitivity = None
    epsilon = None

    q_items = {1: np.array([1, 7]),
               2: np.array([4, 6]),
               6: np.array([4, 6])}
    data = np.array([[1, 1],
                     [1, 2]])
    labels = np.array([1, 6])

    svd_recommender = SVDRecommender(f, lam, g, sensitivity, epsilon)
    svd_recommender._q_items = q_items
    svd_recommender.train(data, labels)

    mu = np.mean(labels)
    n_obs = len(data)

    items_in_data = data[:, 1].tolist()
    q_items_in_data = np.array([q_items[k] for k in items_in_data])

    # USER LATENT FACTORS
    mat = reduce(np.add,
                 [np.outer(q_items_in_data[i], q_items_in_data[i]) for i in range(n_obs)])
    mat = lam * n_obs * np.identity(f) + mat
    vec = reduce(np.add,
                 [q_items_in_data[i] * (labels[i] - mu) for i in range(n_obs)])
    p = np.matmul(np.linalg.inv(mat), vec)

    np.testing.assert_array_equal(svd_recommender._profile, p)

    # GRADIENTS
    grad = [g * ((labels[i] - mu - np.dot(p, q_items_in_data[i])) * p - lam * q_items_in_data[i]) for i in range(n_obs)]
    own_grad = dict(zip(items_in_data, grad))
    for item in items_in_data:
        np.testing.assert_array_equal(svd_recommender._grad[item], own_grad[item])


def test_svd_recommender_predict_wrong_data():
    f = 2
    lam = 0.1
    g = 0.01
    sensitivity = None
    epsilon = None

    q_items = {1: np.array([1, 7]),
               2: np.array([4, 6])}
    data = np.array([[1, 1],
                     [1, 5]])

    svd_recommender = SVDRecommender(f, lam, g, sensitivity, epsilon)
    svd_recommender._q_items = q_items

    with pytest.raises(AssertionError):
        svd_recommender.predict(data)


def test_svd_recommender_predict():
    f = 2
    lam = 0.1
    g = 0.01
    sensitivity = None
    epsilon = None

    q_items = {1: np.array([1, 7]),
               2: np.array([4, 6]),
               6: np.array([4, 6])}
    data = np.array([[1, 1],
                     [1, 2]])

    svd_recommender = SVDRecommender(f, lam, g, sensitivity, epsilon)
    svd_recommender._q_items = q_items
    svd_recommender._mu = 1
    svd_recommender._profile = np.array([2, 4])

    predictions = svd_recommender.predict(data)

    items_in_data = data[:, 1].tolist()
    q_items_in_data = np.array([q_items[k] for k in items_in_data])
    own_predictions = svd_recommender._mu + np.array([np.dot(q_items_in_data[i], svd_recommender._profile)
                                                      for i in range(len(data))])

    np.testing.assert_array_equal(predictions, own_predictions)


def test_svd_recommender_evaluate():
    f = 2
    lam = 0.1
    g = 0.01
    sensitivity = None
    epsilon = None

    q_items = {1: np.array([1, 7]),
               2: np.array([4, 6]),
               6: np.array([4, 6])}
    data = np.array([[1, 1],
                     [1, 2]])
    labels = np.array([1, 6])

    svd_recommender = SVDRecommender(f, lam, g, sensitivity, epsilon)
    svd_recommender._q_items = q_items
    svd_recommender._mu = 1
    svd_recommender._profile = np.array([2, 4])

    predictions = svd_recommender.predict(data)
    rmse_own = np.sqrt(np.mean((predictions - labels) ** 2))
    rmse = svd_recommender.evaluate(data, labels)

    assert rmse == rmse_own


def test_svd_recommender_evaluate_empty_labels():
    f = 2
    lam = 0.1
    g = 0.01
    sensitivity = None
    epsilon = None

    q_items = {1: np.array([1, 7]),
               2: np.array([4, 6]),
               6: np.array([4, 6])}
    data = np.empty((0, 2))
    labels = np.empty(0)

    svd_recommender = SVDRecommender(f, lam, g, sensitivity, epsilon)
    svd_recommender._q_items = q_items
    svd_recommender._mu = 1
    svd_recommender._profile = np.array([2, 4])

    rmse_own = 0
    rmse = svd_recommender.evaluate(data, labels)

    assert rmse == rmse_own


def test_set_model_params():
    f = 2
    lam = 0.1
    g = 0.01
    sensitivity = None
    epsilon = None

    q_items = {1: np.array([1, 7]),
               2: np.array([4, 6]),
               6: np.array([4, 6])}

    svd_recommender = SVDRecommender(f, lam, g, sensitivity, epsilon)
    svd_recommender.set_model_params(q_items)

    np.testing.assert_array_equal(q_items, svd_recommender._q_items)


def test_get_model_params():
    f = 2
    lam = 0.1
    g = 0.01
    sensitivity = None
    epsilon = None

    grad = {1: np.array([1, 7]),
            2: np.array([4, 6]),
            6: np.array([4, 6])}

    svd_recommender = SVDRecommender(f, lam, g, sensitivity, epsilon)
    svd_recommender._grad = grad

    np.testing.assert_array_equal(grad, svd_recommender.get_model_params())


def test_svd_recommender_performance():
    f = 2
    lam = 0.1
    g = 0.01
    sensitivity = None
    epsilon = None

    q_items = {1: np.array([1, 7]),
               2: np.array([4, 6]),
               6: np.array([4, 6])}
    data = np.array([[1, 1],
                     [1, 2]])
    labels = np.array([1, 6])

    svd_recommender = SVDRecommender(f, lam, g, sensitivity, epsilon)
    svd_recommender._q_items = q_items
    svd_recommender._mu = 1
    svd_recommender._profile = np.array([2, 4])

    predictions = svd_recommender.predict(data)
    rmse_own = np.sqrt(np.mean((predictions - labels) ** 2))
    rmse = svd_recommender.performance(data, labels)

    assert rmse == rmse_own


def test_svd_recommender_performance_empty_labels():
    f = 2
    lam = 0.1
    g = 0.01
    sensitivity = None
    epsilon = None

    q_items = {1: np.array([1, 7]),
               2: np.array([4, 6]),
               6: np.array([4, 6])}
    data = np.empty((0, 2))
    labels = np.empty(0)

    svd_recommender = SVDRecommender(f, lam, g, sensitivity, epsilon)
    svd_recommender._q_items = q_items
    svd_recommender._mu = 1
    svd_recommender._profile = np.array([2, 4])

    rmse_own = 0
    rmse = svd_recommender.performance(data, labels)

    assert rmse == rmse_own
