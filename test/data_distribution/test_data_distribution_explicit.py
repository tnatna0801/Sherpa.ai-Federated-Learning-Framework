import numpy as np

from shfl.data_base.data_base import DataBase
from shfl.data_distribution.data_distribution_explicit import ExplicitDataDistribution
from shfl.private import UnprotectedAccess


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


def test_get_data_federated():
    data = TestDataBase()
    data.load_data()
    data_distribution = ExplicitDataDistribution(data)

    federated_data, test_data, test_label = data_distribution.get_federated_data()

    data_access_definition = UnprotectedAccess()
    federated_data.configure_data_access(data_access_definition)
    group_query = federated_data.query()

    for i in range(len(group_query)):
        identifier = federated_data[i].federated_data_identifier
        identifier_data = group_query[i].data[0, 0]
        assert identifier == identifier_data
