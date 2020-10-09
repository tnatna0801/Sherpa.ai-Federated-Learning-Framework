import numpy as np
from unittest.mock import Mock

from shfl.federated_government.federated_recommender_simple import SimpleFederatedRecommender
from shfl.data_base.data_base import DataBase
from shfl.data_distribution.data_distribution_explicit import ExplicitDataDistribution
from shfl.private.data import UnprotectedAccess
from shfl.private.federated_operation import split_train_test


class TestDataBase(DataBase):
    def __init__(self):
        super(TestDataBase, self).__init__()

    def load_data(self):
        self._train_data = np.array([[2, 3, 51],
                                     [1, 34, 6],
                                     [22, 33, 7],
                                     [22, 13, 65],
                                     [1, 3, 15]])
        self._test_data = np.array([[2, 2, 1],
                                    [22, 0, 4],
                                    [1, 1, 5]])
        self._train_labels = np.array([3, 2, 5, 6, 7])
        self._test_labels = np.array([4, 7, 2])


def test_deploy_central_model():
    model_builder = Mock
    database = TestDataBase()
    database.load_data()
    db = ExplicitDataDistribution(database)

    federated_data, test_data, test_labels = db.get_federated_data()

    fdr = SimpleFederatedRecommender(model_builder, federated_data)
    array_params = np.random.rand(30)
    fdr._model.get_model_params.return_value = array_params

    fdr.deploy_central_model()

    for node in fdr._federated_data:
        node._model.set_model_params.assert_called_once()


def test_evaluate_global_model():
    model_builder = Mock
    database = TestDataBase()
    database.load_data()
    db = ExplicitDataDistribution(database)

    federated_data, test_data, test_labels = db.get_federated_data()

    fdr = SimpleFederatedRecommender(model_builder, federated_data)

    for node in fdr._federated_data:
        node.evaluate = Mock()
        node.evaluate.return_value = [np.array(2), None]

    rmse = fdr.evaluate_global_model(test_data, test_labels)
    assert rmse == 2


def test_run_rounds():
    model_builder = Mock
    database = TestDataBase()
    database.load_data()
    db = ExplicitDataDistribution(database)

    federated_data, test_data, test_labels = db.get_federated_data()

    fdr = SimpleFederatedRecommender(model_builder, federated_data)

    for node in fdr._federated_data:
        node.evaluate = Mock()
        node.evaluate.return_value = [np.array(2), None]

    fdr.deploy_central_model()
    fdr.train_all_clients()
    fdr.evaluate_global_model(test_data, test_labels)

    fdr.deploy_central_model = Mock()
    fdr.train_all_clients = Mock()
    fdr.evaluate_global_model = Mock()

    fdr.run_rounds(test_data, test_labels)

    fdr.deploy_central_model.assert_called_once()
    fdr.train_all_clients.assert_called_once()
    fdr.evaluate_global_model.assert_called_once_with(test_data, test_labels)

