import pytest
import numpy as np
import random

from shfl.data_base.data_base import DataBase
from shfl.data_distribution.data_distribution_non_iid import NonIidDataDistribution,choose_labels
from shfl.core.query import Get


class TestDataBase(DataBase):
    def __init__(self):
        super(TestDataBase, self).__init__()

    def load_data(self):
        self._train_data = np.random.rand(250).reshape([50,5])
        self._test_data = np.random.rand(250).reshape([50,5])
        self._validation_data = np.random.rand(250).reshape([50,5])
        self._train_labels = np.random.randint(0,10,50)
        self._test_labels = np.random.randint(0,10,50)
        self._validation_labels = np.random.randint(0,10,50)


def test_choose_labels():
    num_nodes = 3
    total_labels = 10

    random_labels = choose_labels(num_nodes,total_labels)
    all_labels = np.concatenate(random_labels)

    for node in random_labels:
        assert len(node) <= total_labels

    assert len(random_labels) == num_nodes
    assert ((all_labels>=0) & (all_labels<total_labels)).all()


def test_make_data_federated():
    random.seed(123)
    np.random.seed(123)

    data = TestDataBase()
    data.load_data()
    data_distribution = NonIidDataDistribution(data)

    train_data, train_label = data_distribution._database.train
    validation_data, validation_label = data_distribution._database.validation

    train_data = np.concatenate((train_data, validation_data), axis=0)
    train_label = np.concatenate((train_label, validation_label), axis=0)

    num_nodes = 3
    percent = 60
    # weights = np.full(num_nodes, 1/num_nodes)
    weights = [0.5,0.25,0.25]
    federated_data, federated_label = data_distribution.make_data_federated(train_data,
                                                                            train_label,
                                                                            num_nodes,
                                                                            percent,
                                                                            weights)

    all_data = np.concatenate(federated_data)
    all_label = np.concatenate(federated_label)

    idx = []
    for data in all_data:
        idx.append(np.where((data==train_data).all(axis=1))[0][0])

    seed_weights = [6,12,5]
    for i,weight in enumerate(weights):
        assert federated_data[i].shape[0] == seed_weights[i]

    assert all_data.shape[0] == 23
    assert num_nodes == federated_data.shape[0] == federated_label.shape[0]
    assert (np.sort(all_data.ravel()) == np.sort(train_data[idx,].ravel())).all()
    assert (np.sort(all_label) == np.sort(train_label[idx])).all()

def test_get_federated_data():
    random.seed(123)
    np.random.seed(123)

    data = TestDataBase()
    data.load_data()

    dt = NonIidDataDistribution(data)

    # Identifier and num nodes is checked in core test.
    # Percent and weight is checked in idd and no_idd test.
    # So here, we only test mistaken param.
    mistaken = 50
    num_nodes = 4
    federated_data, test_data, test_label = dt.get_federated_data("id001", num_nodes, mistaken=mistaken)

    X_c = []
    y_c = []
    for i in range(federated_data.num_nodes()):
        X_c.append(federated_data[i].query_private_data(Get(), "id001").data)
        y_c.append(federated_data[i].query_private_data(Get(), "id001").label)

    X_c = np.array(X_c)
    y_c = np.array(y_c)

    train_data, train_labels = dt._database.train
    validation_data, validation_labels = dt._database.validation

    X = np.concatenate([train_data, validation_data], axis=0)
    y = np.concatenate([train_labels, validation_labels], axis=0)

    num_mistaken = 0
    idx = []
    for i,node in enumerate(X_c):
        labels_node = []
        for data in node:
            assert data in X
            idx.append(np.where((data==X).all(axis=1))[0][0])
            labels_node.append(y[idx[-1]])
        if not (labels_node == y_c[i]).all():
            num_mistaken = num_mistaken + 1

    assert np.array_equal(X[idx,].ravel(), np.concatenate(X_c).ravel())
    assert np.array_equal(test_data.ravel(), dt._database.test[0].ravel())
    assert np.array_equal(test_label, dt._database.test[1])
    assert num_mistaken / num_nodes * 100 == mistaken
    assert num_mistaken / num_nodes * 100 > 0

