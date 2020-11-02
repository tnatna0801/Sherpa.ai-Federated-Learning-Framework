import numpy as np
import pandas as pd

from shfl.data_base.data_base import DataBase
from shfl.data_distribution.data_distribution_explicit import ExplicitDataDistribution


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
                                     [3, 1, 5]])
        self._train_labels = np.array([3, 2, 5, 6, 7])
        self._test_labels = np.array([4, 7, 2])


class TestDataBasePandas(DataBase):
    def __init__(self):
        super(TestDataBasePandas, self).__init__()

    def load_data(self):
        self._train_data = pd.DataFrame([[2, 3, 51],
                                         [1, 34, 6],
                                         [22, 33, 7],
                                         [22, 13, 65],
                                         [1, 3, 15]])
        self._test_data = pd.DataFrame([[2, 2, 1],
                                        [22, 0, 4],
                                        [3, 1, 5]])
        self._train_labels = pd.DataFrame([3, 2, 5, 6, 7])
        self._test_labels = pd.DataFrame([4, 7, 2])


def test_make_data_federated():
    data = TestDataBase()
    data.load_data()
    data_distribution = ExplicitDataDistribution(data)

    train_data, train_label = data_distribution._database.train

    percent = 100
    federated_data, federated_label = data_distribution.make_data_federated(train_data,
                                                                            train_label,
                                                                            percent)

    all_data = np.concatenate(federated_data)
    all_label = np.concatenate(federated_label)

    idx = []
    for data in all_data:
        idx.append(np.where((data == train_data).all(axis=1))[0][0])

    assert all_data.shape[0] == int(percent * train_data.shape[0] / 100)
    assert len(federated_data) == len(np.unique(train_data[:, 0]))
    assert (np.sort(all_data.ravel()) == np.sort(train_data[idx, ].ravel())).all()
    assert (np.sort(all_label, 0) == np.sort(train_label[idx], 0)).all()


def test_make_data_federated_pandas():
    data = TestDataBasePandas()
    data.load_data()
    data_distribution = ExplicitDataDistribution(data)

    train_data, train_label = data_distribution._database.train

    percent = 100
    federated_data, federated_label = data_distribution.make_data_federated(train_data,
                                                                            train_label,
                                                                            percent)

    all_data = pd.concat(federated_data)
    all_label = pd.concat(federated_label)

    assert all_data.shape[0] == int(percent * train_data.shape[0] / 100)
    assert len(federated_data) == len(np.unique(train_data.iloc[:, 0]))
    pd.testing.assert_frame_equal(all_data, train_data.iloc[all_data.index.values])
    pd.testing.assert_frame_equal(all_label, train_label.iloc[all_data.index.values])
