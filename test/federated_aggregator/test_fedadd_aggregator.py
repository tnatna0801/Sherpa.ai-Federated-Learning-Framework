import numpy as np
import pytest

from shfl.federated_aggregator.fedadd_aggregator import FedAddAggregator


def test_aggregated_weights():
    num_clients = 10
    num_layers = 5
    tams = [[128, 64], [64, 64], [64, 64], [64, 32], [32, 10]]

    weights = []
    for i in range(num_clients):
        weights.append([np.random.rand(tams[j][0], tams[j][1]) for j in range(num_layers)])

    clients_params = np.array(weights)

    avgfa = FedAddAggregator()
    aggregated_weights = avgfa.aggregate_weights(clients_params)

    own_agg = np.array([np.sum(clients_params[:, layer], axis=0) for layer in range(num_layers)])

    for i in range(num_layers):
        assert np.array_equal(own_agg[i], aggregated_weights[i])
    assert aggregated_weights.shape[0] == num_layers


def test_aggregated_weights_multidimensional_2d_array():
    num_clients = 10
    num_rows_params = 3
    num_cols_params = 9
    
    clients_params = []
    for i in range(num_clients):
        clients_params.append(np.random.rand(num_rows_params, num_cols_params))

    avgfa = FedAddAggregator()
    aggregated_weights = avgfa.aggregate_weights(clients_params)
    
    own_agg = np.zeros((num_rows_params, num_cols_params))
    for i_client in range(num_clients): 
        own_agg += clients_params[i_client]

    assert np.array_equal(own_agg, aggregated_weights)
    assert aggregated_weights.shape == own_agg.shape


def test_aggregated_weights_multidimensional_3d_array():
    num_clients = 10
    num_rows_params = 3
    num_cols_params = 9
    num_k_params = 5
    
    clients_params = []
    for i in range(num_clients):
        clients_params.append(np.random.rand(num_rows_params, num_cols_params, num_k_params))

    avgfa = FedAddAggregator()
    aggregated_weights = avgfa.aggregate_weights(clients_params)
    
    own_agg = np.zeros((num_rows_params, num_cols_params, num_k_params))
    for i_client in range(num_clients): 
        own_agg += clients_params[i_client]

    assert np.array_equal(own_agg, aggregated_weights)
    assert aggregated_weights.shape == own_agg.shape


def test_aggregated_weights_list():
    num_clients = 10
    num_list_1 = 3
    num_list_2 = 9

    clients_params = []
    for i in range(num_clients):
        clients_params.append([np.random.rand(num_list_1), np.random.rand(num_list_2)])


    avgfa = FedAddAggregator()
    aggregated_weights = avgfa.aggregate_weights(clients_params)

    own_agg = [np.zeros(num_list_1), np.zeros(num_list_2)]
    for i_client in range(num_clients):
        own_agg[0] += clients_params[i_client][0]
        own_agg[1] += clients_params[i_client][1]

    assert np.array_equal(own_agg[0], aggregated_weights[0])


def test_aggregated_weights_dict():
    clients_params = [{0: np.array([1, 3, 4]),
                       1: np.array([1, 2, 3])},
                      {0: np.array([3, 5, 7]),
                       2: np.array([1, 1, 6])}]

    avgfa = FedAddAggregator()
    aggregated_weights = avgfa.aggregate_weights(clients_params)

    own_agg = {0: np.array([4, 8, 11]),
               1: np.array([1, 2, 3]),
               2: np.array([1, 1, 6])}

    for k in aggregated_weights.keys():
        np.testing.assert_equal(aggregated_weights[k], own_agg[k])


def test_aggregated_weights_wrong_type():
    clients_params = [(1, 2)]

    avgfa = FedAddAggregator()

    with pytest.raises(TypeError):
        avgfa.aggregate_weights(clients_params)
