import numpy as np
from functools import reduce

from shfl.federated_aggregator.federated_aggregator import FederatedAggregator


class FedAddAggregator(FederatedAggregator):
    """
    Implementation of an Additive Federated Aggregator. It only uses a simple sum of the parameters of all the models.

    It implements [Federated Aggregator](../federated_aggregator/#federatedaggregator-class)
    """

    @staticmethod
    def _check_client_params(client_params):
        if not isinstance(client_params[0], (np.ScalarType, np.ndarray, list, dict)):
            raise TypeError(
                "The client parameters must be either a (list of) scalar/ndarray or a dictionary.")

    @staticmethod
    def _dict_reducer(accumulator, element):
        for key, value in element.items():
            accumulator[key] = accumulator.get(key, 0) + value
        return accumulator

    def aggregate_weights(self, clients_params):
        """
        Implementation of abstract method of class
        [AggregateWeightsFunction](../federated_aggregator/#federatedaggregator-class)
        # Arguments:
            clients_params: list of arrays or dictionaries. Each entry in the list contains the model's parameters
            of one client.

        # Returns
            aggregated_weights: aggregator weights

        """
        self._check_client_params(clients_params)
        if isinstance(clients_params[0], dict):
            aggregated_weights = reduce(self._dict_reducer, clients_params)
        elif isinstance(clients_params[0], list):
            aggregated_weights = [np.sum(params, axis=0) for params in zip(*clients_params)]
        else:
            aggregated_weights = np.sum(clients_params, axis=0)

        return aggregated_weights
